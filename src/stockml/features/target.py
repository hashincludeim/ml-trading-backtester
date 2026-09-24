"""Prediction targets and the no-model baseline for the volatility target."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockml.config import TargetConfig

TARGET_NAME = "price_rise"
VOLATILITY_TARGET_NAME = "big_move"


def make_price_rise_target(close: pd.Series) -> pd.Series:
    """Binary target: 1 if the next close is higher than today's close, else 0.

    The value at row ``t`` depends on ``close[t+1]``, so it is a *label*, never a feature. The
    final row has no next close; it is returned as ``<NA>`` and must be dropped before training.

    Args:
        close: Close prices indexed by date.

    Returns:
        Nullable ``Int8`` Series named ``price_rise``.
    """
    next_close = close.shift(-1)
    target = (next_close > close).astype("Int8")
    return target.mask(next_close.isna()).rename(TARGET_NAME)


def _absolute_returns(close: pd.Series) -> pd.Series:
    return pd.Series(np.log(close / close.shift(1)), index=close.index).abs()


def typical_absolute_return(close: pd.Series, window: int) -> pd.Series:
    """Median absolute daily log return over the trailing ``window`` days, including day ``t``.

    Uses closes up to ``t`` only. NaN until ``window`` returns are available.
    """
    return _absolute_returns(close).rolling(window, min_periods=window).median()


def make_volatility_target(close: pd.Series, window: int) -> pd.Series:
    """Binary target: 1 if tomorrow's absolute log return beats the trailing typical move.

    The threshold at row ``t`` (:func:`typical_absolute_return`) uses closes up to ``t``; only
    the compared move, ``|log(close[t+1] / close[t])|``, looks ahead, which makes it a label.
    Rows without a full threshold window, and the final row, are ``<NA>``.

    Args:
        close: Close prices indexed by date.
        window: Trailing days that define a typical absolute move.

    Returns:
        Nullable ``Int8`` Series named ``big_move``.
    """
    next_move = _absolute_returns(close).shift(-1)
    threshold = typical_absolute_return(close, window)
    target = (next_move > threshold).astype("Int8")
    return target.mask(next_move.isna() | threshold.isna()).rename(VOLATILITY_TARGET_NAME)


def make_target(close: pd.Series, config: TargetConfig | None = None) -> pd.Series:
    """Build the target named by ``config.kind`` (direction by default)."""
    cfg = config or TargetConfig()
    if cfg.kind == "volatility":
        return make_volatility_target(close, cfg.volatility_window)
    return make_price_rise_target(close)


def volatility_persistence_score(close: pd.Series, config: TargetConfig) -> pd.Series:
    """No-model baseline for the volatility target: recent moves relative to a typical move.

    Mean absolute return over the last ``persistence_window`` days divided by the trailing
    typical move. Values above 1 mean "the last few days were rougher than usual", so the rule
    predicts a big move. Uses closes up to ``t`` only.

    Args:
        close: Close prices indexed by date.
        config: Supplies ``persistence_window`` and ``volatility_window``.

    Returns:
        Score Series aligned with ``close`` (NaN during warm-up).
    """
    recent = _absolute_returns(close).rolling(config.persistence_window).mean()
    typical = typical_absolute_return(close, config.volatility_window)
    return (recent / typical.replace(0.0, np.nan)).rename("persistence_score")
