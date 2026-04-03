from __future__ import annotations

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def classification_metrics(df: pd.DataFrame, y_col: str, p_col: str) -> dict[str, float]:
    y = df[y_col].astype(int)
    p = df[p_col].astype(float).clip(1e-6, 1 - 1e-6)
    return {
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "mean_predicted_prob": float(p.mean()),
    }
