"""NRCL-87 synthetic fixtures shaped by Kalshi's September 6, 2026 docs.

These are not captured production responses or evidence of authenticated access.
"""

import json
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, localcontext

import pytest
import requests

from neural.kalshi import (
    BASE_URL,
    KalshiDataClient,
    KalshiDataError,
    parse_market,
    parse_orderbook,
    snapshot_json,
)

NOW = datetime(2026, 9, 6, 12, tzinfo=timezone.utc)


@pytest.fixture
def market():
    return {
        "ticker": "KX-EXAMPLE-YES",
        "event_ticker": "KX-EXAMPLE",
        "market_type": "binary",
        "title": "Example event",
        "status": "active",
        "yes_bid_dollars": "0.0000",
        "yes_ask_dollars": "0.4001",
        "no_bid_dollars": "0.5999",
        "no_ask_dollars": "1.0000",
        "last_price_dollars": "0.4000",
        "volume_fp": "123456789012345.25",
        "price_ranges": [{"start": "0.0000", "end": "1.0000", "step": "0.0001"}],
        "exchange_index": 2,
        "updated_time": "2026-09-06T11:00:00Z",
        "future_metadata": {"ignored": True},
    }


@pytest.fixture
def book():
    return {
        "orderbook_fp": {
            "yes_dollars": [["0.1000", "2.50"], ["0.3001", "1.25"]],
            "no_dollars": [["0.5999", "4.75"]],
        }
    }


class Response:
    def __init__(self, payload, status=200, headers=None):
        self.payload, self.status_code, self.headers = payload, status, headers or {}
        self.closed = False

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return deepcopy(self.payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def close(self):
        self.closed = True


class Session:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, deepcopy(kwargs)))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.closed = True


def test_exact_decimal_market_and_serialization(market):
    snapshot = parse_market(market, received_at=NOW)
    assert snapshot.yes_bid == Decimal("0")
    assert snapshot.yes_ask == Decimal("0.4001")
    assert snapshot.volume == Decimal("123456789012345.25")
    assert snapshot.price_ranges[0].step == Decimal("0.0001")
    assert snapshot.exchange_index == 2
    assert snapshot.status == "active"
    assert snapshot.updated_at < snapshot.received_at
    wire = json.loads(snapshot_json(snapshot))
    assert wire["yes_ask"] == "0.4001"
    assert wire["volume"] == "123456789012345.25"
    assert wire["venue"] == "kalshi" and wire["schema_version"] == "1.0.0"
    assert "future_metadata" not in wire
    market["yes_bid_dollars"] = None
    assert parse_market(market, received_at=NOW).yes_bid is None


@pytest.mark.parametrize(
    "value",
    [1, 0.4, True, "NaN", "Infinity", "1e-2", "-0.1", "1.0001", "0.12345", "0.4\n", "", "00.4"],
)
def test_invalid_price_fails_closed(market, value):
    market["yes_bid_dollars"] = value
    with pytest.raises(KalshiDataError):
        parse_market(market, received_at=NOW)


@pytest.mark.parametrize(
    "field,value",
    [
        ("volume_fp", "1.001"),
        ("market_type", "scalar"),
        ("exchange_index", True),
        ("exchange_index", -1),
        ("updated_time", "2026-09-06T12:00:00"),
        ("updated_time", "yesterday"),
        ("price_ranges", None),
        ("title", ""),
        ("mve_collection_ticker", "COMBO"),
        ("ticker", "../portfolio"),
    ],
)
def test_bad_market_metadata(market, field, value):
    market[field] = value
    with pytest.raises(KalshiDataError):
        parse_market(market, received_at=NOW)


def test_legacy_fields_not_used(market):
    market["yes_bid"] = 99
    assert parse_market(market, received_at=NOW).yes_bid == 0
    del market["yes_bid_dollars"]
    with pytest.raises(KalshiDataError, match="legacy"):
        parse_market(market, received_at=NOW)


@pytest.mark.parametrize(
    "bands",
    [
        [{"start": "0", "end": "1", "step": "0"}],
        [{"start": "1", "end": "0", "step": "0.01"}],
        [{"start": "0", "end": "1"}],
        [
            {"start": "0", "end": "0.6", "step": "0.01"},
            {"start": "0.5", "end": "1", "step": "0.01"},
        ],
    ],
)
def test_bad_price_ranges(market, bands):
    market["price_ranges"] = bands
    with pytest.raises(KalshiDataError):
        parse_market(market, received_at=NOW)


def test_book_depth_and_complements_preserve_precision(book):
    snapshot = parse_orderbook(book, ticker="KX-EXAMPLE", received_at=NOW)
    assert [level.price for level in snapshot.yes_bids] == [Decimal("0.3001"), Decimal("0.1")]
    with localcontext() as context:
        context.prec = 2
        assert snapshot.asks("yes")[0].price == Decimal("0.4001")
        assert snapshot.asks("no")[0].price == Decimal("0.6999")
    assert snapshot.asks("yes")[0].quantity == Decimal("4.75")
    wire = json.loads(snapshot_json(snapshot))
    assert wire["yes_bids"][0]["quantity"] == "1.25"
    assert "updated_at" not in wire  # no invented exchange timestamp
    with pytest.raises(ValueError):
        snapshot.asks("maybe")
    book["orderbook_fp"]["yes_dollars"] = []
    assert parse_orderbook(book, ticker="KX-EXAMPLE", received_at=NOW).asks("no") == ()


@pytest.mark.parametrize(
    "rows",
    [
        None,
        [["0.1"]],
        [[0.1, "1"]],
        [["0.1", 1]],
        [["0.1", "0"]],
        [["0.1", "1.001"]],
        [["0.1", "1"], ["0.1000", "2"]],
    ],
)
def test_malformed_book(book, rows):
    book["orderbook_fp"]["yes_dollars"] = rows
    with pytest.raises(KalshiDataError):
        parse_orderbook(book, ticker="KX-EXAMPLE", received_at=NOW)


def test_missing_book_not_empty_liquidity():
    for payload in [{}, {"orderbook": {"yes": [[10, 5]]}}, {"orderbook_fp": {}}]:
        with pytest.raises(KalshiDataError):
            parse_orderbook(payload, ticker="KX-EXAMPLE", received_at=NOW)


def test_explicit_routes_and_signer_only_on_book(market, book):
    responses = [Response({"market": market}), Response(book)]
    session = Session(*responses)
    signed = []

    def sign(method, path):
        signed.append((method, path))
        return {"KALSHI-ACCESS-KEY": "test-only"}

    with KalshiDataClient(session=session, signer_headers=sign, clock=lambda: NOW) as client:
        assert client.get_market(market["ticker"]).received_at == NOW
        assert client.get_orderbook(market["ticker"], depth=5).requested_depth == 5
    assert signed == [("GET", "/trade-api/v2/markets/KX-EXAMPLE-YES/orderbook")]
    assert session.calls[0][0] == BASE_URL + "/markets/KX-EXAMPLE-YES"
    assert session.calls[0][1]["headers"] == {}
    assert session.calls[1][1]["params"] == {"depth": 5}
    assert all(call[1]["allow_redirects"] is False for call in session.calls)
    assert all(response.closed for response in responses)
    assert not session.closed  # injected transport belongs to caller
    with pytest.raises(RuntimeError, match="closed"):
        client.get_market(market["ticker"])


def test_default_session_does_not_load_environment_credentials(monkeypatch):
    session = Session()
    monkeypatch.setattr(requests, "Session", lambda: session)
    with KalshiDataClient():
        assert session.trust_env is False
    assert session.closed


def test_paginated_generic_discovery(market):
    session = Session(
        Response({"markets": [market], "cursor": "next"}),
        Response({"markets": [market], "cursor": ""}),
    )
    client = KalshiDataClient(session=session, clock=lambda: NOW)
    assert len(list(client.iter_markets(page_size=1, max_pages=2))) == 2
    assert session.calls[0][1]["params"] == {"limit": 1, "mve_filter": "exclude", "status": "open"}
    assert session.calls[1][1]["params"]["cursor"] == "next"


@pytest.mark.parametrize(
    "response,max_pages,error",
    [
        ({"markets": [], "cursor": "next"}, 1, "incomplete"),
        ({"markets": [], "cursor": "next"}, 2, "repeated"),
        ({"markets": []}, 2, "cursor"),
        ({"cursor": ""}, 2, "markets"),
    ],
)
def test_pagination_never_silently_truncates(response, max_pages, error):
    session = Session(Response(response), Response(response))
    with pytest.raises(KalshiDataError, match=error):
        list(KalshiDataClient(session=session).iter_markets(max_pages=max_pages))


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_bounded_retries(market, monkeypatch, status):
    delays = []
    monkeypatch.setattr("neural.kalshi.sleep", delays.append)
    failed = Response({}, status, {"Retry-After": "1"})
    session = Session(failed, Response({"market": market}))
    KalshiDataClient(session=session).get_market(market["ticker"])
    assert len(session.calls) == 2 and delays == [1.0] and failed.closed


@pytest.mark.parametrize(
    "status,retry_after", [(401, None), (404, None), (429, "120"), (503, "tomorrow")]
)
def test_no_unauthorized_or_early_retry(status, retry_after):
    response = Response({}, status, {} if retry_after is None else {"Retry-After": retry_after})
    session = Session(response)
    with pytest.raises(requests.HTTPError):
        KalshiDataClient(session=session).get_market("KX-EXAMPLE")
    assert len(session.calls) == 1 and response.closed


def test_network_retries_exhaust_and_malformed_json_closes(monkeypatch):
    monkeypatch.setattr("neural.kalshi.sleep", lambda _: None)
    session = Session(requests.Timeout(), requests.Timeout(), requests.Timeout())
    with pytest.raises(requests.Timeout):
        KalshiDataClient(session=session).get_market("KX-EXAMPLE")
    assert len(session.calls) == 3
    response = Response(ValueError("bad JSON"))
    with pytest.raises(ValueError, match="bad JSON"):
        KalshiDataClient(session=Session(response)).get_market("KX-EXAMPLE")
    assert response.closed


def test_bad_identity_and_redirect(market):
    with pytest.raises(KalshiDataError, match="match"):
        KalshiDataClient(session=Session(Response({"market": market}))).get_market("OTHER")
    with pytest.raises(KalshiDataError, match="redirect"):
        KalshiDataClient(session=Session(Response({}, 302))).get_market("KX-EXAMPLE")


def test_input_limits_before_network():
    client = KalshiDataClient(session=Session())
    for value in [-1, 101, True]:
        with pytest.raises(ValueError):
            client.get_orderbook("KX-EXAMPLE", depth=value)
    for kwargs in [{"page_size": 0}, {"page_size": True}, {"max_pages": 0}, {"status": "wrong"}]:
        with pytest.raises(ValueError):
            list(client.iter_markets(**kwargs))
    for kwargs in [
        {"timeout": float("nan")},
        {"timeout": 0},
        {"max_retries": True},
        {"max_retries": 4},
    ]:
        with pytest.raises(ValueError):
            KalshiDataClient(**kwargs)
