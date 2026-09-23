"""Validation and outlier handling for daily price data."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from stockml.config import PRICE_COLUMNS, DataConfig

logger = logging.getLogger(__name__)


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Check and tidy a raw OHLCV frame.

    Sorts by date, drops duplicate dates and rows with missing or non-positive prices, and fixes
    ``High``/``Low`` so they bracket ``Open`` and ``Close``.

    Args:
        prices: Frame with ``PRICE_COLUMNS`` indexed by date.

    Returns:
        A new, cleaned frame.

    Raises:
        ValueError: If required columns are missing or no valid rows remain.
    """
    missing = [c for c in PRICE_COLUMNS if c not in prices.columns]
    if missing:
        raise ValueError(f"Missing price columns: {missing}")
    df = prices.loc[:, list(PRICE_COLUMNS)].copy()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    ohlc = ["Open", "High", "Low", "Close"]
    df = df.dropna(subset=ohlc)
    df = df[(df[ohlc] > 0).all(axis=1)]
    df["Volume"] = df["Volume"].fillna(0.0).clip(lower=0.0)
    df["High"] = df[ohlc].max(axis=1)
    df["Low"] = df[ohlc].min(axis=1)
    dropped = len(prices) - len(df)
    if dropped:
        logger.info("validate_prices dropped %d invalid rows", dropped)
    if df.empty:
        raise ValueError("No valid price rows remain after validation")
    return df


def detect_outliers(close: pd.Series, config: DataConfig | None = None) -> pd.Series:
    """Flag one-day bad ticks from extreme daily log returns.

    Two definitions of "extreme" are supported (``config.outlier_method``):

    * ``"zscore"``: ``|r_t| / std(r_{t-w..t-1}) > threshold``. Scaling by trailing volatility
      stops genuine crisis-era moves from being flagged.
    * ``"iqr"``: ``r_t`` outside ``[Q1 - k*IQR, Q3 + k*IQR]`` with ``IQR = Q3 - Q1``.

    With ``outlier_require_reversal`` a day is only flagged when its return *and* the next
    return are both extreme with opposite signs (a spike that snaps back). This looks one day
    ahead, which is fine for repairing historical data but means this function must never be
    used to build model features.

    Args:
        close: Close prices indexed by date.
        config: Outlier settings.

    Returns:
        Boolean Series aligned with ``close``; ``True`` marks an outlier price.
    """
    cfg = config or DataConfig()
    returns = np.log(close).diff()
    if cfg.outlier_method == "iqr":
        q1, q3 = returns.quantile(0.25), returns.quantile(0.75)
        iqr = q3 - q1
        extreme = (returns < q1 - cfg.outlier_iqr_k * iqr) | (
            returns > q3 + cfg.outlier_iqr_k * iqr
        )
    else:
        window = cfg.outlier_zscore_window
        trailing_vol = returns.rolling(window, min_periods=max(2, window // 3)).std().shift(1)
        extreme = (returns / trailing_vol).abs() > cfg.outlier_zscore_threshold
    extreme = pd.Series(extreme.fillna(False).astype(bool), index=close.index)
    if not cfg.outlier_require_reversal:
        return extreme
    next_extreme = extreme.shift(-1, fill_value=False).astype(bool)
    reverses = np.sign(returns) == -np.sign(returns.shift(-1))
    return pd.Series(extreme & next_extreme & reverses, index=close.index, dtype=bool)


def replace_outliers(prices: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Replace flagged rows' OHLC with the previous valid close.

    Args:
        prices: OHLCV frame.
        mask: Boolean Series from :func:`detect_outliers`, aligned with ``prices``.

    Returns:
        A new frame where each flagged row's Open/High/Low/Close equals the prior close.
    """
    df = prices.copy()
    flagged = mask.reindex(df.index, fill_value=False).astype(bool)
    if not flagged.any():
        return df
    prev_close = df["Close"].mask(flagged).ffill().shift(1)
    for col in ("Open", "High", "Low", "Close"):
        df.loc[flagged, col] = prev_close[flagged]
    logger.info("Replaced %d outlier rows: %s", int(flagged.sum()), list(df.index[flagged]))
    return df.dropna(subset=["Close"])


def clean_prices(prices: pd.DataFrame, config: DataConfig | None = None) -> pd.DataFrame:
    """Validate prices and repair one-day outliers in a single call."""
    cfg = config or DataConfig()
    df = validate_prices(prices)
    mask = detect_outliers(df["Close"], cfg)
    return replace_outliers(df, mask)
