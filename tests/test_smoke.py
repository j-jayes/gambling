from __future__ import annotations

import pandas as pd

from worldcup_markets.backtest.engine import run_simple_backtest
from worldcup_markets.portfolio.sizing import fractional_kelly


def test_fractional_kelly_bounded() -> None:
    stake = fractional_kelly(prob=0.6, odds_decimal=2.2, fraction=0.5, cap=0.03)
    assert 0.0 <= stake <= 0.03


def test_backtest_returns_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "market_prob_mean": [0.45, 0.55, 0.40],
            "pred_prob_class_1": [0.55, 0.62, 0.41],
            "target": [1, 0, 1],
        }
    )
    out = run_simple_backtest(df)
    assert {"edge", "signal", "pnl_unit", "cum_pnl_unit"}.issubset(out.columns)
