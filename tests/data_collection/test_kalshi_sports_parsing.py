from __future__ import annotations

import pandas as pd
import pytest

from neural.data_collection import kalshi


def test_parse_market_date_from_ticker_uses_yy_mmm_dd() -> None:
    parsed = kalshi._parse_market_date_from_ticker("KXNBAGAME-26MAR10CHIGSW-GSW")

    assert parsed == pd.Timestamp("2026-03-10")


def test_parse_matchup_teams_supports_title_at_winner_format() -> None:
    row = pd.Series({"title": "Chicago at Golden State Winner?", "subtitle": ""})

    teams = kalshi._parse_matchup_teams(row)

    assert teams.to_dict() == {"home_team": "Golden State", "away_team": "Chicago"}


@pytest.mark.asyncio
async def test_get_nba_games_enriches_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = pd.DataFrame(
        [
            {
                "ticker": "KXNBAGAME-26MAR10CHIGSW-GSW",
                "title": "Chicago at Golden State Winner?",
                "subtitle": "",
                "yes_ask": 70,
                "volume": 0,
            }
        ]
    )

    async def fake_get_markets_by_sport(*args, **kwargs):
        return sample

    monkeypatch.setattr(kalshi, "get_markets_by_sport", fake_get_markets_by_sport)

    result = await kalshi.get_nba_games(use_authenticated=False)

    assert len(result) == 1
    assert result.iloc[0]["home_team"] == "Golden State"
    assert result.iloc[0]["away_team"] == "Chicago"
    assert result.iloc[0]["game_date"] == pd.Timestamp("2026-03-10")


@pytest.mark.asyncio
async def test_get_nfl_games_enriches_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = pd.DataFrame(
        [
            {
                "ticker": "KXNFLGAME-26SEP22DETBAL-BAL",
                "title": "Will the Detroit beat the Baltimore?",
                "subtitle": "",
            }
        ]
    )

    async def fake_get_markets_by_sport(*args, **kwargs):
        return sample

    monkeypatch.setattr(kalshi, "get_markets_by_sport", fake_get_markets_by_sport)

    result = await kalshi.get_nfl_games(use_authenticated=False)

    assert len(result) == 1
    assert result.iloc[0]["home_team"] == "Baltimore"
    assert result.iloc[0]["away_team"] == "Detroit"
    assert result.iloc[0]["game_date"] == pd.Timestamp("2026-09-22")


@pytest.mark.asyncio
async def test_get_cfb_games_enriches_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = pd.DataFrame(
        [
            {
                "ticker": "KXNCAAFGAME-26NOV28MICHOSU-OSU",
                "title": "Michigan at Ohio State Winner?",
                "subtitle": "",
            }
        ]
    )

    seen: dict[str, object] = {}

    async def fake_get_markets_by_sport(*args, **kwargs):
        seen.update(kwargs)
        return sample

    monkeypatch.setattr(kalshi, "get_markets_by_sport", fake_get_markets_by_sport)

    result = await kalshi.get_cfb_games(use_authenticated=False)

    assert seen["sport"] == "CFB"
    assert len(result) == 1
    assert result.iloc[0]["home_team"] == "Ohio State"
    assert result.iloc[0]["away_team"] == "Michigan"
    assert result.iloc[0]["game_date"] == pd.Timestamp("2026-11-28")
