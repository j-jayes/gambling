from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseSourceClient(ABC):
    @abstractmethod
    def fetch(self, **kwargs: object) -> pd.DataFrame:
        """Fetch source data and return a normalized DataFrame."""


class BaseOddsClient(BaseSourceClient):
    pass


class BaseStatsClient(BaseSourceClient):
    pass
