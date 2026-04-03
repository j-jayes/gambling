from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests

from worldcup_markets.data_sources.base import BaseOddsClient


class TheOddsApiClient(BaseOddsClient):
    def __init__(self, api_key: str, base_url: str = "https://api.the-odds-api.com/v4", timeout: int = 20):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def fetch(
        self,
        sport_key: str,
        regions: str = "eu,uk,us",
        markets: str = "h2h,totals",
        odds_format: str = "decimal",
    ) -> pd.DataFrame:
        endpoint = f"{self.base_url}/sports/{sport_key}/odds"
        response = requests.get(
            endpoint,
            params={
                "apiKey": self.api_key,
                "regions": regions,
                "markets": markets,
                "oddsFormat": odds_format,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        rows: list[dict[str, object]] = []
        now = datetime.now(timezone.utc)
        for event in payload:
            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        price = outcome.get("price")
                        implied_prob = (1.0 / float(price)) if price and price > 0 else None
                        rows.append(
                            {
                                "source": "bookmaker",
                                "source_event_id": event.get("id"),
                                "source_market_id": f"{bookmaker.get('key')}::{market.get('key')}",
                                "event_name": f"{event.get('home_team')} vs {event.get('away_team')}",
                                "market_name": market.get("key"),
                                "selection_name": outcome.get("name"),
                                "market_type": "1x2" if market.get("key") == "h2h" else "other",
                                "quote_ts_utc": now,
                                "event_start_ts_utc": event.get("commence_time"),
                                "decimal_odds": price,
                                "implied_prob_raw": implied_prob,
                                "implied_prob_vig_adj": None,
                                "overround": None,
                                "currency": "USD",
                                "metadata": {
                                    "bookmaker": bookmaker.get("key"),
                                    "region": regions,
                                },
                            }
                        )

        return pd.DataFrame(rows)
