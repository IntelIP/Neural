from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_example_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "examples" / "13_deterministic_cross_provider_replay.py"
    spec = importlib.util.spec_from_file_location("deterministic_cross_provider_replay", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deterministic_replay_produces_expected_signals():
    module = _load_example_module()

    stream = module.build_replay_stream()
    snapshot = module.latest_snapshot(stream)
    analysis = module.analyze_standardized_stream(stream)

    assert not stream.empty
    assert not snapshot.empty
    assert not analysis.empty

    signals = {
        row["market_id"]: row["signal"]
        for row in analysis[["market_id", "signal"]].to_dict("records")
    }

    assert signals["KXNBAGAME-26MAR10CHINYK-NYK"] == "buy_yes"
    assert signals["pm-bos-lal-moneyline"] == "buy_no"
    assert signals["pm-den-dal-moneyline"] == "hold"

    assert set(snapshot["exchange"]) == {"kalshi", "polymarket_us"}
