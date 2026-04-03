from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO

import pandas as pd
import requests

from worldcup_markets.data_sources.base import BaseOddsClient


class FootballDataCoUkClient(BaseOddsClient):
    """Open historical odds from football-data.co.uk.

    Uses the public mmz4281 season CSV layout.
    """

    def __init__(self, base_url: str = "https://www.football-data.co.uk/mmz4281", timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def fetch(self, season_codes: list[str], division: str = "WC") -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for season_code in season_codes:
            url = f"{self.base_url}/{season_code}/{division}.csv"
            response = requests.get(url, timeout=self.timeout)
            if response.status_code != 200:
                continue

            text = response.content.decode("cp1252", errors="replace")
            raw = pd.read_csv(StringIO(text))
            if raw.empty:
                continue

            normalized = self._normalize(raw, season_code, division)
            if not normalized.empty:
                frames.append(normalized)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _normalize(self, raw: pd.DataFrame, season_code: str, division: str) -> pd.DataFrame:
        required = {"Date", "HomeTeam", "AwayTeam", "FTR"}
        if not required.issubset(set(raw.columns)):
            return pd.DataFrame()

        odds_triplets = [
            ("B365H", "B365D", "B365A"),
            ("PSH", "PSD", "PSA"),
            ("WHH", "WHD", "WHA"),
            ("AvgH", "AvgD", "AvgA"),
        ]

        used_cols: tuple[str, str, str] | None = None
        for home_col, draw_col, away_col in odds_triplets:
            if {home_col, draw_col, away_col}.issubset(set(raw.columns)):
                used_cols = (home_col, draw_col, away_col)
                break

        if not used_cols:
            return pd.DataFrame()

        home_col, draw_col, away_col = used_cols
        rows: list[dict[str, object]] = []
        ingest_ts = datetime.now(timezone.utc)

        for idx, match in raw.iterrows():
            home = str(match.get("HomeTeam", "")).strip()
            away = str(match.get("AwayTeam", "")).strip()
            date_val = pd.to_datetime(match.get("Date"), dayfirst=True, errors="coerce")
            if not home or not away or pd.isna(date_val):
                continue

            event_name = f"{home} vs {away}"
            event_ts = date_val.to_pydatetime().replace(tzinfo=timezone.utc)
            source_event_id = f"{season_code}:{idx}:{home}:{away}"
            result_code = str(match.get("FTR", "")).strip().upper()

            selections = [
                ("Home", match.get(home_col), result_code == "H"),
                ("Draw", match.get(draw_col), result_code == "D"),
                ("Away", match.get(away_col), result_code == "A"),
            ]

            for selection_name, price, did_win in selections:
                price_float = pd.to_numeric(price, errors="coerce")
                if pd.isna(price_float) or float(price_float) <= 1.0:
                    continue

                implied = 1.0 / float(price_float)
                rows.append(
                    {
                        "source": "bookmaker",
                        "source_event_id": source_event_id,
                        "source_market_id": f"football-data::{season_code}::{division}::h2h",
                        "event_name": event_name,
                        "market_name": "h2h",
                        "selection_name": selection_name,
                        "market_type": "1x2",
                        "quote_ts_utc": ingest_ts,
                        "event_start_ts_utc": event_ts,
                        "decimal_odds": float(price_float),
                        "implied_prob_raw": implied,
                        "implied_prob_vig_adj": None,
                        "overround": None,
                        "currency": "N/A",
                        "target": int(did_win),
                        "metadata": {
                            "season_code": season_code,
                            "division": division,
                            "result_code": result_code,
                            "odds_columns": {
                                "home": home_col,
                                "draw": draw_col,
                                "away": away_col,
                            },
                        },
                    }
                )

        return pd.DataFrame(rows)