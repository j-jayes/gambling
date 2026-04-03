from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@dataclass
class TrainedModel:
    pipeline: Pipeline
    feature_columns: list[str]


def train_classifier(df: pd.DataFrame, target_col: str) -> TrainedModel:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    feature_columns = [c for c in df.columns if c not in [target_col, "event_name"]]
    if not feature_columns:
        raise ValueError("No feature columns available for training")

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]
    )

    X = df[feature_columns]
    y = df[target_col]
    pipeline.fit(X, y)
    return TrainedModel(pipeline=pipeline, feature_columns=feature_columns)
