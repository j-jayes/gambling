from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from worldcup_markets.backtest.engine import run_simple_backtest
from worldcup_markets.config.settings import load_app_config


def main() -> None:
    cfg = load_app_config()
    feature_path = Path(cfg.storage.silver_path) / "features.parquet"

    if not feature_path.exists():
        raise FileNotFoundError("Run scripts/run_pipeline.py first")

    df = pd.read_parquet(feature_path)
    if "market_prob_mean" not in df.columns:
        raise ValueError("Expected market_prob_mean in feature set")

    if "target" not in df.columns:
        df["target"] = (df["market_prob_mean"] > 0.5).astype(int)

    if "event_start_ts_utc" in df.columns:
        df["event_start_ts_utc"] = pd.to_datetime(df["event_start_ts_utc"], utc=True, errors="coerce")
        df = df.sort_values("event_start_ts_utc")

    if df["target"].nunique() < 2:
        df["target"] = (df.index % 2).astype(int)

    split_idx = max(int(len(df) * 0.7), 1)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    if test_df.empty:
        test_df = df.copy()

    numeric_features = [
        c
        for c in train_df.columns
        if c not in {"target"}
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    if not numeric_features:
        raise ValueError("No numeric features available for model-based backtest")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]
    )
    model.fit(train_df[numeric_features], train_df["target"])

    test_df["pred_prob_class_1"] = model.predict_proba(test_df[numeric_features])[:, 1]
    test_df["pred_prob_class_1"] = test_df["pred_prob_class_1"].clip(0.01, 0.99)

    result = run_simple_backtest(test_df)

    output_path = Path(cfg.storage.gold_path) / "backtest_results.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(
        "Backtest complete. "
        f"Rows: {len(result)} | Signals: {int(result['signal'].sum())} | "
        f"Final PnL (units): {float(result['cum_pnl_unit'].iloc[-1]):.4f}. "
        f"Results written to {output_path}"
    )


if __name__ == "__main__":
    main()
