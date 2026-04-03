from __future__ import annotations

import pandas as pd


def run_simple_backtest(
    predictions: pd.DataFrame,
    market_prob_col: str = "market_prob_mean",
    model_prob_col: str = "pred_prob_class_1",
    edge_threshold: float = 0.02,
) -> pd.DataFrame:
    if predictions.empty:
        return predictions

    out = predictions.copy()
    out["edge"] = out[model_prob_col] - out[market_prob_col]
    out["signal"] = (out["edge"] >= edge_threshold).astype(int)
    if "target" in out.columns:
        out["pnl_unit"] = out["signal"] * ((out["target"] * (1 / out[market_prob_col])) - 1)
    else:
        out["pnl_unit"] = 0.0
    out["cum_pnl_unit"] = out["pnl_unit"].cumsum()
    return out
