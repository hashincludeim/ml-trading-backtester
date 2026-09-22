"""Descriptive analytics behind the dashboard's explanatory charts.

These are pure functions over prices, returns, features and predictions. They describe what
happened; none of them feed model training. Anything computed over the full history (for
example feature bucket statistics) includes the test period and is for exploration only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockml.config import TRADING_DAYS_PER_YEAR, FeatureConfig
from stockml.features.technical import log_returns

MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


@dataclass(frozen=True)
class DrawdownEpisode:
    """The deepest peak-to-trough fall of a price or equity series."""

    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None
    depth: float  # positive fraction, e.g. 0.93 = 93% below the peak


def calendar_returns(log_ret: pd.Series) -> pd.DataFrame:
    """Simple returns per calendar month, as a year × month table.

    Args:
        log_ret: Daily log returns indexed by date.

    Returns:
        Frame indexed by year with columns ``Jan`` … ``Dec``; months with no data are NaN.
    """
    r = log_ret.dropna()
    idx = pd.DatetimeIndex(r.index)
    monthly = r.groupby([idx.year, idx.month]).sum()
    table: pd.DataFrame = np.expm1(monthly).unstack().reindex(columns=range(1, 13))
    table.columns = list(MONTH_LABELS)
    table.index.name = "year"
    return table


def annual_returns(log_ret: pd.Series) -> pd.Series:
    """Simple return per calendar year from daily log returns."""
    r = log_ret.dropna()
    yearly = r.groupby(pd.DatetimeIndex(r.index).year).sum()
    return pd.Series(np.expm1(yearly), index=yearly.index, name="annual_return")


def rolling_annualised_volatility(
    close: pd.Series, window: int, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> pd.Series:
    """Trailing-window standard deviation of daily log returns, annualised."""
    vol = log_returns(close).rolling(window, min_periods=window).std()
    return pd.Series(vol * np.sqrt(periods_per_year), index=close.index, name="volatility")


def worst_drawdown(series: pd.Series) -> DrawdownEpisode:
    """Locate the deepest drawdown and when (if ever) the prior peak was regained."""
    running_peak = series.cummax()
    dd = series / running_peak - 1.0
    trough = pd.Timestamp(str(dd.idxmin()))
    peak = pd.Timestamp(str(series.loc[:trough].idxmax()))
    after = series.loc[trough:]
    recovered = after[after >= series.loc[peak]]
    return DrawdownEpisode(
        peak_date=peak,
        trough_date=trough,
        recovery_date=pd.Timestamp(recovered.index[0]) if len(recovered) else None,
        depth=float(-dd.min()),
    )


def autocorrelation(returns: pd.Series, max_lag: int) -> pd.Series:
    """Sample autocorrelation of ``returns`` at lags ``1..max_lag``."""
    r = returns.dropna()
    return pd.Series(
        [r.autocorr(lag) for lag in range(1, max_lag + 1)],
        index=pd.RangeIndex(1, max_lag + 1, name="lag"),
        name="acf",
    )


def noise_band(n: int, z: float = 1.96) -> float:
    """Approximate 95% band for a correlation estimated from ``n`` independent observations."""
    return float(z / np.sqrt(max(n, 1)))


def feature_bucket_stats(feature: pd.Series, y: pd.Series, n_bins: int) -> pd.DataFrame:
    """Next-day up-rate within quantile buckets of one feature.

    Args:
        feature: Feature values.
        y: Binary target aligned with ``feature``.
        n_bins: Requested number of quantile buckets (fewer if values repeat).

    Returns:
        Frame with one row per bucket: ``lower``, ``upper``, ``median`` (feature value),
        ``up_rate``, ``se`` (binomial standard error) and ``n``.
    """
    buckets = pd.qcut(feature, q=n_bins, duplicates="drop")
    grouped = pd.DataFrame({"x": feature, "y": y, "b": buckets}).groupby("b", observed=True)
    stats = grouped.agg(
        lower=("x", "min"),
        upper=("x", "max"),
        median=("x", "median"),
        up_rate=("y", "mean"),
        n=("y", "size"),
    )
    stats["se"] = np.sqrt(stats["up_rate"] * (1 - stats["up_rate"]) / stats["n"])
    return stats.reset_index(drop=True)


def feature_return_correlation(X: pd.DataFrame, forward_returns: pd.Series) -> pd.Series:
    """Spearman rank correlation of each feature with the next-day return, sorted by size."""
    aligned = forward_returns.reindex(X.index)
    corr = X.corrwith(aligned, method="spearman")
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def indicator_signal_table(
    indicators: pd.DataFrame, config: FeatureConfig | None = None
) -> pd.DataFrame:
    """Next-day up-rate after common indicator states (RSI zones, bands, EMA trend, MACD).

    Args:
        indicators: Output of :func:`stockml.features.pipeline.compute_indicators`.
        config: Supplies RSI thresholds and window names.

    Returns:
        Frame with ``indicator``, ``state``, ``up_rate``, ``se``, ``n`` and
        ``mean_return`` (average next-day log return), one row per state with data.
    """
    cfg = config or FeatureConfig()
    close = indicators["Close"]
    nxt = pd.Series(np.log(close.shift(-1) / close), index=close.index)
    up = (nxt > 0).astype(float).where(nxt.notna())
    short, long = (
        indicators[f"ema_{cfg.ema_cross_short}"],
        indicators[f"ema_{cfg.ema_cross_long}"],
    )
    rsi = indicators["rsi"]
    states: list[tuple[str, str, pd.Series]] = [
        ("RSI", f"Oversold (<{cfg.rsi_oversold:g})", rsi < cfg.rsi_oversold),
        ("RSI", "Neutral", (rsi >= cfg.rsi_oversold) & (rsi <= cfg.rsi_overbought)),
        ("RSI", f"Overbought (>{cfg.rsi_overbought:g})", rsi > cfg.rsi_overbought),
        ("Bollinger", "Below lower band", close < indicators["bb_lower"]),
        (
            "Bollinger",
            "Inside bands",
            (close >= indicators["bb_lower"]) & (close <= indicators["bb_upper"]),
        ),
        ("Bollinger", "Above upper band", close > indicators["bb_upper"]),
        ("EMA trend", f"Bullish ({cfg.ema_cross_short} > {cfg.ema_cross_long})", short > long),
        ("EMA trend", f"Bearish ({cfg.ema_cross_short} < {cfg.ema_cross_long})", short < long),
        ("MACD", "Histogram > 0", indicators["macd_hist"] > 0),
        ("MACD", "Histogram < 0", indicators["macd_hist"] < 0),
    ]
    rows = []
    for family, state, mask in states:
        sel = mask.fillna(False).astype(bool) & up.notna()
        n = int(sel.sum())
        if n == 0:
            continue
        p = float(up[sel].mean())
        rows.append(
            {
                "indicator": family,
                "state": state,
                "up_rate": p,
                "se": float(np.sqrt(p * (1 - p) / n)),
                "n": n,
                "mean_return": float(nxt[sel].mean()),
            }
        )
    return pd.DataFrame(rows)


def rolling_hit_rate(y_true: pd.Series, y_pred: pd.Series, window: int) -> pd.Series:
    """Share of correct predictions over a trailing window."""
    correct = (y_true.astype(int) == y_pred.astype(int)).astype(float)
    return correct.rolling(window, min_periods=window).mean()


def binned_density(
    values: pd.Series | np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram density on fixed edges; returns ``(bin_centres, density)``."""
    density, _ = np.histogram(np.asarray(values, dtype=float), bins=edges, density=True)
    return (edges[:-1] + edges[1:]) / 2, density


def shared_edges(frames: Sequence[pd.Series], bins: int, clip: float = 0.005) -> np.ndarray:
    """Common histogram edges covering the central ``1 - 2*clip`` mass of all series."""
    pooled = pd.concat([s.dropna() for s in frames])
    lo, hi = pooled.quantile([clip, 1 - clip])
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return np.linspace(float(lo), float(hi), bins + 1)
