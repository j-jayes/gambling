from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


MarketType = Literal["1x2", "outright", "totals", "exact_score", "other"]
SourceType = Literal["polymarket", "bookmaker", "stats", "model"]


class QuoteRecord(BaseModel):
    source: SourceType
    source_event_id: str
    source_market_id: str
    event_name: str
    market_name: str
    selection_name: str
    market_type: MarketType = "other"
    quote_ts_utc: datetime
    event_start_ts_utc: datetime | None = None
    decimal_odds: float | None = None
    implied_prob_raw: float | None = None
    implied_prob_vig_adj: float | None = None
    overround: float | None = None
    currency: str | None = "USD"
    metadata: dict[str, object] = Field(default_factory=dict)


class MatchFeatureRow(BaseModel):
    event_id: str
    event_start_ts_utc: datetime
    home_team: str
    away_team: str
    elo_home: float | None = None
    elo_away: float | None = None
    xg_home_rolling_5: float | None = None
    xg_away_rolling_5: float | None = None
    rest_days_home: float | None = None
    rest_days_away: float | None = None
    travel_km_home: float | None = None
    travel_km_away: float | None = None
    market_prob_home: float | None = None
    market_prob_draw: float | None = None
    market_prob_away: float | None = None
    y_home_win: int | None = None
    y_draw: int | None = None
    y_away_win: int | None = None
