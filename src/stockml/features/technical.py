"""Pure technical-indicator functions.

Every function takes a Series (or frame) and returns a new object. The value at row ``t`` uses
only data up to and including row ``t`` (trailing windows, no centred windows, no negative
shifts), so all outputs are safe to use as features known at the close of day ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    """Daily log return ``ln(C_t / C_{t-1})``; the first value is NaN."""
    return pd.Series(np.log(close / close.shift(1)), index=close.index, name="log_return")


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over the trailing ``window`` rows (NaN until the window fills)."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with ``span`` (``adjust=False``, recursive form).

    The first ``span - 1`` values are NaN so the average is warmed up before it is used.
    """
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rolling_volatility(close: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation of daily log returns (not annualised)."""
    return log_returns(close).rolling(window=window, min_periods=window).std()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    Args:
        close: Close prices.
        window: Look-back length ``n``; smoothing uses ``alpha = 1 / n``.

    Returns:
        RSI in ``[0, 100]``; NaN for the first ``window`` rows. A window with no losses gives 100.
    """
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss != 0, 100.0).where(avg_gain.notna())
    return out.rename("rsi")


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Moving Average Convergence Divergence.

    Args:
        close: Close prices.
        fast: Span of the fast EMA.
        slow: Span of the slow EMA.
        signal: Span of the signal-line EMA of the MACD line.

    Returns:
        Frame with columns ``macd`` (fast EMA - slow EMA), ``signal``, and ``hist``
        (``macd - signal``).
    """
    line = ema(close, fast) - ema(close, slow)
    signal_line = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": signal_line, "hist": line - signal_line})


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands around a simple moving average.

    Args:
        close: Close prices.
        window: SMA / standard deviation window.
        num_std: Band half-width in standard deviations.

    Returns:
        Frame with ``middle``, ``upper``, ``lower`` and ``position``, where ``position`` is
        ``(close - middle) / (upper - lower)``: 0 at the middle band, +/-0.5 at the bands.
    """
    middle = sma(close, window)
    std = close.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {"middle": middle, "upper": upper, "lower": lower, "position": (close - middle) / width}
    )


def ema_crossovers(short_ema: pd.Series, long_ema: pd.Series) -> pd.DataFrame:
    """Detect bullish/bearish EMA crossover events.

    Args:
        short_ema: Faster EMA.
        long_ema: Slower EMA.

    Returns:
        Boolean frame with ``bullish`` (short crosses above long on this row) and ``bearish``
        (short crosses below long on this row).
    """
    above = short_ema > long_ema
    below = short_ema < long_ema
    prev_above = above.shift(1, fill_value=False).astype(bool)
    prev_below = below.shift(1, fill_value=False).astype(bool)
    return pd.DataFrame({"bullish": above & prev_below, "bearish": below & prev_above})
