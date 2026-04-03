from __future__ import annotations

import pandas as pd

from worldcup_markets.models.train import TrainedModel


def predict_probabilities(model: TrainedModel, df: pd.DataFrame) -> pd.DataFrame:
    X = df[model.feature_columns]
    probs = model.pipeline.predict_proba(X)

    out = df.copy()
    out["pred_prob_class_0"] = probs[:, 0]
    if probs.shape[1] > 1:
        out["pred_prob_class_1"] = probs[:, 1]
    return out
