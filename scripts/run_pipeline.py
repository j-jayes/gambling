from __future__ import annotations

from pathlib import Path
import logging
import json

import pandas as pd

from worldcup_markets.config.settings import Secrets, load_app_config
from worldcup_markets.data_engineering.storage import upsert_to_duckdb, write_parquet
from worldcup_markets.data_sources.odds.football_data_co_uk import FootballDataCoUkClient
from worldcup_markets.data_sources.odds.polymarket import PolymarketGammaClient
from worldcup_markets.data_sources.odds.the_odds_api import TheOddsApiClient
from worldcup_markets.data_sources.stats.soccerdata_adapter import SoccerDataFBrefClient
from worldcup_markets.features.build_features import build_match_features


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_app_config()
    secrets = Secrets()

    polymarket_df = PolymarketGammaClient().fetch(limit=200)

    football_data_df = FootballDataCoUkClient().fetch(
        season_codes=["1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425"],
        division="WC",
    )

    odds_df = pd.DataFrame()
    if secrets.the_odds_api_key:
        odds_df = TheOddsApiClient(api_key=secrets.the_odds_api_key).fetch(
            sport_key="soccer_fifa_world_cup_winner"
        )

    stats_df = SoccerDataFBrefClient(
        leagues=["INT-World Cup"],
        seasons=["2018", "2022"],
    ).fetch()
    if stats_df.empty:
        logging.warning("No stats rows available from soccerdata. Continuing with odds-only features.")

    quote_frames = [polymarket_df, football_data_df]
    if not odds_df.empty:
        quote_frames.append(odds_df)

    non_empty_frames = [q for q in quote_frames if not q.empty]
    if not non_empty_frames:
        raise RuntimeError("No quote data fetched from open or configured sources")

    quotes = pd.concat(non_empty_frames, ignore_index=True)
    if "quote_ts_utc" in quotes.columns:
        quotes["quote_ts_utc"] = pd.to_datetime(quotes["quote_ts_utc"], utc=True, errors="coerce")
    if "event_start_ts_utc" in quotes.columns:
        quotes["event_start_ts_utc"] = pd.to_datetime(quotes["event_start_ts_utc"], utc=True, errors="coerce")
    if "metadata" in quotes.columns:
        quotes["metadata"] = quotes["metadata"].apply(
            lambda v: json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v)
        )

    features_df = build_match_features(quotes, stats_df)

    bronze_path = Path(cfg.storage.bronze_path) / "quotes.parquet"
    silver_path = Path(cfg.storage.silver_path) / "features.parquet"

    write_parquet(quotes, bronze_path)
    write_parquet(features_df, silver_path)

    upsert_to_duckdb(quotes, cfg.storage.db_path, "quotes")
    upsert_to_duckdb(features_df, cfg.storage.db_path, "features")

    print(f"Wrote {len(quotes)} quotes and {len(features_df)} feature rows")


if __name__ == "__main__":
    main()
