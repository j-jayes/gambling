from __future__ import annotations


def fractional_kelly(prob: float, odds_decimal: float, fraction: float = 0.5, cap: float = 0.03) -> float:
    """Return bounded fractional Kelly stake fraction."""
    b = odds_decimal - 1.0
    q = 1.0 - prob
    raw = ((b * prob) - q) / b if b > 0 else 0.0
    stake = max(0.0, raw * fraction)
    return min(stake, cap)
