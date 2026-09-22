"""Prediction target."""

from __future__ import annotations

import pandas as pd

TARGET_NAME = "price_rise"


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
