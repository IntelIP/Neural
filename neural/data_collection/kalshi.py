async def get_nfl_games(
    status: str = "open",
    limit: int = 50,
    use_authenticated: bool = True,
    api_key_id: Optional[str] = None,
    private_key_pem: Optional[bytes] = None,
) -> pd.DataFrame:
    """
    Get NFL games markets from Kalshi.

    Args:
        status: Market status filter (default: 'open')
        limit: Maximum markets to fetch (default: 50)
        use_authenticated: Use authenticated API
        api_key_id: Optional API key
        private_key_pem: Optional private key

    Returns:
        DataFrame with NFL markets, including parsed teams and game date
    """
    df = await get_markets_by_sport(
        sport="NFL",
        status=status,
        limit=limit,
        use_authenticated=use_authenticated,
        api_key_id=api_key_id,
        private_key_pem=private_key_pem,
    )

    if not df.empty:
        # Parse teams from title (common format: "Will the [Away] beat the [Home]?" or similar)
        def parse_teams(row):
            title = row["title"]
            match = re.search(
                r"Will the (\w+(?:\s\w+)?) beat the (\w+(?:\s\w+)?)\?", title, re.IGNORECASE
            )
            if match:
                away, home = match.groups()
                return pd.Series({"home_team": home, "away_team": away})
            # Fallback: extract from subtitle or ticker
            subtitle = row.get("subtitle", "")
            if " vs " in subtitle:
                teams = subtitle.split(" vs ")
                return pd.Series(
                    {
                        "home_team": teams[1].strip() if len(teams) > 1 else None,
                        "away_team": teams[0].strip(),
                    }
                )
            return pd.Series({"home_team": None, "away_team": None})

        team_df = df.apply(parse_teams, axis=1)
        df = pd.concat([df, team_df], axis=1)

        # Parse game date from ticker (format: KXNFLGAME-25SEP22DETBAL -> 25SEP22)
        def parse_game_date(ticker):
            match = re.search(r"-(\d{2}[A-Z]{3}\d{2})", ticker)
            if match:
                date_str = match.group(1)
                try:
                    # Assume YYMMMDD, convert to full year (e.g., 22 -> 2022)
                    year = (
                        int(date_str[-2:]) + 2000
                        if int(date_str[-2:]) < 50
                        else 1900 + int(date_str[-2:])
                    )
                    month_map = {
                        "JAN": 1,
                        "FEB": 2,
                        "MAR": 3,
                        "APR": 4,
                        "MAY": 5,
                        "JUN": 6,
                        "JUL": 7,
                        "AUG": 8,
                        "SEP": 9,
                        "OCT": 10,
                        "NOV": 11,
                        "DEC": 12,
                    }
                    month = month_map.get(date_str[2:5])
                    day = int(date_str[0:2])
                    return pd.to_datetime(f"{year}-{month:02d}-{day:02d}")
                except:
                    pass
            return row.get("open_time", pd.NaT)

        df["game_date"] = df["ticker"].apply(parse_game_date)

        # Filter to ensure NFL-specific
        nfl_mask = df["series_ticker"].str.contains("KXNFLGAME", na=False) | df[
            "title"
        ].str.contains("NFL", case=False, na=False)
        df = df[nfl_mask]

    return df


async def get_cfb_games(
    status: str = "open",
    limit: int = 50,
    use_authenticated: bool = True,
    api_key_id: Optional[str] = None,
    private_key_pem: Optional[bytes] = None,
) -> pd.DataFrame:
    """
    Get College Football (CFB) games markets from Kalshi.

    Args:
        status: Market status filter (default: 'open')
        limit: Maximum markets to fetch (default: 50)
        use_authenticated: Use authenticated API
        api_key_id: Optional API key
        private_key_pem: Optional private key

    Returns:
        DataFrame with CFB markets, including parsed teams and game date
    """
    df = await get_markets_by_sport(
        sport="NCAA Football",
        status=status,
        limit=limit,
        use_authenticated=use_authenticated,
        api_key_id=api_key_id,
        private_key_pem=private_key_pem,
    )

    if not df.empty:
        # Parse teams similar to NFL
        def parse_teams(row):
            title = row["title"]
            match = re.search(
                r"Will the (\w+(?:\s\w+)?) beat the (\w+(?:\s\w+)?)\?", title, re.IGNORECASE
            )
            if match:
                away, home = match.groups()
                return pd.Series({"home_team": home, "away_team": away})
            subtitle = row.get("subtitle", "")
            if " vs " in subtitle:
                teams = subtitle.split(" vs ")
                return pd.Series(
                    {
                        "home_team": teams[1].strip() if len(teams) > 1 else None,
                        "away_team": teams[0].strip(),
                    }
                )
            return pd.Series({"home_team": None, "away_team": None})

        team_df = df.apply(parse_teams, axis=1)
        df = pd.concat([df, team_df], axis=1)

        # Parse game date from ticker
        def parse_game_date(ticker):
            match = re.search(r"-(\d{2}[A-Z]{3}\d{2})", ticker)
            if match:
                date_str = match.group(1)
                try:
                    year = (
                        int(date_str[-2:]) + 2000
                        if int(date_str[-2:]) < 50
                        else 1900 + int(date_str[-2:])
                    )
                    month_map = {
                        "JAN": 1,
                        "FEB": 2,
                        "MAR": 3,
                        "APR": 4,
                        "MAY": 5,
                        "JUN": 6,
                        "JUL": 7,
                        "AUG": 8,
                        "SEP": 9,
                        "OCT": 10,
                        "NOV": 11,
                        "DEC": 12,
                    }
                    month = month_map.get(date_str[2:5])
                    day = int(date_str[0:2])
                    return pd.to_datetime(f"{year}-{month:02d}-{day:02d}")
                except:
                    pass
            return row.get("open_time", pd.NaT)

        df["game_date"] = df["ticker"].apply(parse_game_date)

        # Filter to ensure CFB-specific
        cfb_mask = df["series_ticker"].str.contains("KXNCAAFGAME", na=False) | df[
            "title"
        ].str.contains("NCAA|College Football", case=False, na=False)
        df = df[cfb_mask]

    return df
