from __future__ import annotations

import numpy as np
import pandas as pd


def build_match_features(quotes_df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    """Build minimal feature set to bootstrap modeling.

    This is intentionally simple and should be replaced with richer tournament features.
    """
    if quotes_df.empty:
        return pd.DataFrame()

    quotes = quotes_df.copy()
    quotes["event_name"] = quotes["event_name"].astype(str)

    # Prefer 1X2 odds rows when available because they map naturally to match-level labels.
    if "market_type" in quotes.columns:
        odds_1x2 = quotes[quotes["market_type"].eq("1x2")].copy()
    else:
        odds_1x2 = pd.DataFrame()
    selection_map = {"Home": "home", "Draw": "draw", "Away": "away"}

    if not odds_1x2.empty and "selection_name" in odds_1x2.columns:
        odds_1x2["selection_key"] = odds_1x2["selection_name"].map(selection_map)
        odds_1x2 = odds_1x2[odds_1x2["selection_key"].notna()]

        if not odds_1x2.empty:
            agg = (
                odds_1x2.groupby(["source_event_id", "event_name", "selection_key"], dropna=False)
                .agg(
                    implied_prob_raw=("implied_prob_raw", "mean"),
                    decimal_odds=("decimal_odds", "mean"),
                    event_start_ts_utc=("event_start_ts_utc", "max"),
                    target=("target", "max"),
                )
                .reset_index()
            )

            pivot_prob = (
                agg.pivot(index=["source_event_id", "event_name"], columns="selection_key", values="implied_prob_raw")
                .rename(columns={
                    "home": "market_prob_home_raw",
                    "draw": "market_prob_draw_raw",
                    "away": "market_prob_away_raw",
                })
                .reset_index()
            )

            pivot_odds = (
                agg.pivot(index=["source_event_id", "event_name"], columns="selection_key", values="decimal_odds")
                .rename(columns={
                    "home": "home_odds_mean",
                    "draw": "draw_odds_mean",
                    "away": "away_odds_mean",
                })
                .reset_index()
            )

            event_ts = (
                agg.groupby(["source_event_id", "event_name"], dropna=False)["event_start_ts_utc"]
                .max()
                .reset_index()
            )

            target_home = (
                agg[agg["selection_key"] == "home"]
                .groupby(["source_event_id", "event_name"], dropna=False)["target"]
                .max()
                .reset_index()
                .rename(columns={"target": "target"})
            )

            features = pivot_prob.merge(pivot_odds, on=["source_event_id", "event_name"], how="left")
            features = features.merge(event_ts, on=["source_event_id", "event_name"], how="left")
            features = features.merge(target_home, on=["source_event_id", "event_name"], how="left")

            for col in ["market_prob_home_raw", "market_prob_draw_raw", "market_prob_away_raw"]:
                features[col] = pd.to_numeric(features[col], errors="coerce")

            features["overround"] = (
                features["market_prob_home_raw"].fillna(0.0)
                + features["market_prob_draw_raw"].fillna(0.0)
                + features["market_prob_away_raw"].fillna(0.0)
            )
            features["overround"] = features["overround"].replace(0.0, np.nan)

            features["market_prob_home"] = (features["market_prob_home_raw"] / features["overround"]).fillna(0.0)
            features["market_prob_draw"] = (features["market_prob_draw_raw"] / features["overround"]).fillna(0.0)
            features["market_prob_away"] = (features["market_prob_away_raw"] / features["overround"]).fillna(0.0)

            features["market_prob_mean"] = features["market_prob_home"].clip(0.001, 0.999)
            features["market_prob_std"] = features[["market_prob_home", "market_prob_draw", "market_prob_away"]].std(axis=1)
            features["feature_stub"] = np.linspace(0.0, 1.0, num=len(features), endpoint=False)

            if "event_start_ts_utc" in features.columns:
                features["event_start_ts_utc"] = pd.to_datetime(features["event_start_ts_utc"], utc=True, errors="coerce")

            # Keep target as the home-win label to align with backtest assumptions.
            if "target" in features.columns:
                features["target"] = pd.to_numeric(features["target"], errors="coerce").fillna(0).astype(int)

            return features

    market_probs = (
        quotes.groupby("event_name")
        .agg(
            market_prob_mean=("implied_prob_raw", "mean"),
            market_prob_std=("implied_prob_raw", "std"),
        )
        .reset_index()
    )
    market_probs["market_prob_mean"] = market_probs["market_prob_mean"].fillna(0.5).astype(float)
    market_probs["market_prob_std"] = market_probs["market_prob_std"].fillna(0.0).astype(float)

    if stats_df.empty or "game" not in stats_df.columns:
        market_probs["feature_stub"] = np.linspace(0.0, 1.0, num=len(market_probs), endpoint=False)
        return market_probs

    stats = stats_df.copy()
    stats["event_name"] = stats["game"].astype(str)
    merged = market_probs.merge(stats[["event_name"]], on="event_name", how="left")
    merged["feature_stub"] = np.linspace(0.0, 1.0, num=len(merged), endpoint=False)
    return merged
