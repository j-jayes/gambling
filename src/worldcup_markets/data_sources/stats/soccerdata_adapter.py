from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd
import soccerdata as sd

from worldcup_markets.data_sources.base import BaseStatsClient


class SoccerDataFBrefClient(BaseStatsClient):
    def __init__(self, leagues: list[str], seasons: list[str], data_dir: str = "data/external/soccerdata_cache"):
        self.leagues = leagues
        self.seasons = seasons
        self.data_dir = Path(data_dir)

    def fetch(self, **kwargs: object) -> pd.DataFrame:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            fbref = sd.FBref(leagues=self.leagues, seasons=self.seasons, data_dir=self.data_dir)
            schedule = fbref.read_schedule()
            if isinstance(schedule.index, pd.MultiIndex):
                schedule = schedule.reset_index()
            return schedule
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                "soccerdata FBref fetch failed, continuing with empty stats frame: %s", exc
            )
            return pd.DataFrame()
