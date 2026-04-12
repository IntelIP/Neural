import pytest

from neural.trading.fix_streaming import FIXStreamingClient


def test_handle_message_reports_fix_parse_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[str] = []
    client = FIXStreamingClient(on_error=errors.append)

    monkeypatch.setattr(
        "neural.trading.fix_streaming.KalshiFIXClient.to_dict",
        lambda message: (_ for _ in ()).throw(ValueError("bad fix payload")),
    )

    client._handle_message(object())  # type: ignore[arg-type]

    assert errors == ["Error parsing FIX message: bad fix payload"]


def test_market_data_snapshot_reports_parse_errors() -> None:
    errors: list[str] = []
    client = FIXStreamingClient(on_error=errors.append)

    client._handle_market_data_snapshot(
        {
            55: "KX-1",
            132: "bad-price",
            133: "101",
            134: "1",
            135: "1",
        }
    )

    assert errors == ["Error parsing market data snapshot for KX-1: could not convert string to float: 'bad-price'"]
    assert client.get_snapshot("KX-1") is None


def test_market_data_snapshot_propagates_callback_errors() -> None:
    client = FIXStreamingClient(
        on_market_data=lambda snapshot: (_ for _ in ()).throw(ValueError("bad callback"))
    )

    with pytest.raises(ValueError, match="bad callback"):
        client._handle_market_data_snapshot(
            {
                55: "KX-1",
                132: "55",
                133: "57",
                134: "10",
                135: "12",
            }
        )

    snapshot = client.get_snapshot("KX-1")
    assert snapshot is not None
    assert snapshot.symbol == "KX-1"
