from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests

from worldcup_markets.data_sources.base import BaseOddsClient


class PolymarketGammaClient(BaseOddsClient):
    def __init__(self, base_url: str = "https://gamma-api.polymarket.com", timeout: int = 20) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def fetch(self, active: bool = True, closed: bool = False, limit: int = 100) -> pd.DataFrame:
        endpoint = f"{self.base_url}/markets"
        response = requests.get(
            endpoint,
            params={"active": str(active).lower(), "closed": str(closed).lower(), "limit": limit},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        rows: list[dict[str, object]] = []
        now = datetime.now(timezone.utc)
        for market in payload:
            token_ids = market.get("clobTokenIds") or []
            if isinstance(token_ids, str):
                token_ids = [t.strip() for t in token_ids.split(",") if t.strip()]

            rows.append(
                {
                    "source": "polymarket",
                    "source_event_id": str(market.get("eventId", "")),
                    "source_market_id": str(market.get("id", "")),
                    "event_name": market.get("question", ""),
                    "market_name": market.get("question", ""),
                    "selection_name": "yes/no",
                    "market_type": "other",
                    "quote_ts_utc": now,
                    "event_start_ts_utc": market.get("endDate"),
                    "decimal_odds": None,
                    "implied_prob_raw": market.get("probability"),
                    "implied_prob_vig_adj": market.get("probability"),
                    "overround": None,
                    "currency": "USD",
                    "metadata": {
                        "slug": market.get("slug"),
                        "token_ids": token_ids,
                    },
                }
            )

        return pd.DataFrame(rows)
