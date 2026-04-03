from __future__ import annotations

from pathlib import Path

import pandas as pd

from worldcup_markets.config.settings import load_app_config
from worldcup_markets.models.train import train_classifier


def main() -> None:
    cfg = load_app_config()
    feature_path = Path(cfg.storage.silver_path) / "features.parquet"

    if not feature_path.exists():
        raise FileNotFoundError("Run scripts/run_pipeline.py first")

    df = pd.read_parquet(feature_path)
    if "target" not in df.columns:
        # Bootstrap target for pipeline smoke-testing; replace with real labels.
        if "market_prob_mean" in df.columns:
            df["target"] = (df["market_prob_mean"] > 0.5).astype(int)
        else:
            df["target"] = 0

    if df["target"].nunique() < 2:
        # Keep the scaffold executable even when upstream labels are unavailable.
        df["target"] = (df.index % 2).astype(int)

    model = train_classifier(df, target_col="target")

    Path("data/gold").mkdir(parents=True, exist_ok=True)
    pd.Series(model.feature_columns, name="feature").to_csv("data/gold/model_features.csv", index=False)
    print("Model trained. Feature list written to data/gold/model_features.csv")


if __name__ == "__main__":
    main()
