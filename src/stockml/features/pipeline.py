"""Assemble indicators and model features from a price frame."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from stockml.config import FeatureConfig
from stockml.features.target import make_price_rise_target
from stockml.features.technical import (
    bollinger,
    ema,
    ema_crossovers,
    log_returns,
    macd,
    rolling_volatility,
    rsi,
    sma,
)

logger = logging.getLogger(__name__)


def compute_indicators(prices: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Compute price-level indicators used for charts and as inputs to features.

    Args:
        prices: OHLCV frame.
        config: Indicator windows.

    Returns:
        A new frame with the original OHLCV columns plus ``sma_{w}``, ``ema_{w}``, ``bb_*``,
        ``rsi``, ``macd``, ``macd_signal``, ``macd_hist``, ``ema_cross_bullish`` and
        ``ema_cross_bearish``. Values at row ``t`` use data up to ``t`` only.
    """
    cfg = config or FeatureConfig()
    close = prices["Close"]
    out = prices.copy()
    ema_spans = sorted({*cfg.ema_windows, cfg.ema_cross_short, cfg.ema_cross_long})
    for w in cfg.sma_windows:
        out[f"sma_{w}"] = sma(close, w)
    for w in ema_spans:
        out[f"ema_{w}"] = ema(close, w)
    bands = bollinger(close, cfg.bollinger_window, cfg.bollinger_num_std)
    for col in bands.columns:
        out[f"bb_{col}"] = bands[col]
    out["rsi"] = rsi(close, cfg.rsi_window)
    m = macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    out["macd"] = m["macd"]
    out["macd_signal"] = m["signal"]
    out["macd_hist"] = m["hist"]
    crosses = ema_crossovers(out[f"ema_{cfg.ema_cross_short}"], out[f"ema_{cfg.ema_cross_long}"])
    out["ema_cross_bullish"] = crosses["bullish"]
    out["ema_cross_bearish"] = crosses["bearish"]
    return out


def compute_features(prices: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Build the full (unfiltered) feature table named by ``config.feature_columns``.

    Level indicators are expressed relative to the close so features are roughly stationary.
    Rows during indicator warm-up contain NaN.

    Args:
        prices: OHLCV frame.
        config: Feature settings.

    Returns:
        Frame whose columns are exactly ``config.feature_columns``, indexed like ``prices``.
    """
    cfg = config or FeatureConfig()
    ind = compute_indicators(prices, cfg)
    close = prices["Close"]
    feats = pd.DataFrame(index=prices.index)
    feats["hl_range"] = (prices["High"] - prices["Low"]) / close
    feats["oc_change"] = (close - prices["Open"]) / prices["Open"]
    feats["return_1d"] = log_returns(close)
    for w in cfg.sma_windows:
        feats[f"sma_{w}_ratio"] = close / ind[f"sma_{w}"] - 1.0
    for w in cfg.ema_windows:
        feats[f"ema_{w}_ratio"] = close / ind[f"ema_{w}"] - 1.0
    feats[f"volatility_{cfg.volatility_window}"] = rolling_volatility(close, cfg.volatility_window)
    short_ema, long_ema = ind[f"ema_{cfg.ema_cross_short}"], ind[f"ema_{cfg.ema_cross_long}"]
    feats["ema_bullish"] = (short_ema > long_ema).astype(float).where(long_ema.notna())
    feats["ema_diff_pct"] = (short_ema - long_ema) / close
    feats[f"rsi_{cfg.rsi_window}"] = ind["rsi"]
    feats["bb_position"] = ind["bb_position"]
    feats["macd_pct"] = ind["macd"] / close
    feats["macd_hist_pct"] = ind["macd_hist"] / close
    avg_volume = prices["Volume"].rolling(cfg.volume_window, min_periods=cfg.volume_window).mean()
    feats["volume_ratio"] = prices["Volume"] / avg_volume.replace(0.0, np.nan)
    feats = feats.replace([np.inf, -np.inf], np.nan)
    return feats.loc[:, list(cfg.feature_columns)]


def build_feature_frame(
    prices: pd.DataFrame, config: FeatureConfig | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Return model-ready ``(X, y)`` with warm-up rows and the final unlabeled row removed.

    Args:
        prices: Cleaned OHLCV frame indexed by date.
        config: Feature settings.

    Returns:
        ``X`` with columns ``config.feature_columns`` and ``y`` (int, 1 = next close higher),
        sharing the same chronologically sorted index. Row ``t`` of ``X`` uses information
        available at the close of day ``t`` only.
    """
    cfg = config or FeatureConfig()
    feats = compute_features(prices, cfg)
    target = make_price_rise_target(prices["Close"])
    valid = feats.notna().all(axis=1) & target.notna()
    X = feats.loc[valid]
    y = target.loc[valid].astype(int)
    logger.debug("build_feature_frame: %d usable rows of %d", len(X), len(prices))
    return X, y
