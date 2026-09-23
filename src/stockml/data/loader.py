"""Download daily OHLCV prices from Yahoo Finance and cache them as Parquet."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import yfinance as yf

from stockml.config import PRICE_COLUMNS, DataConfig

logger = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._^-]")  # keep "^" so index symbols round-trip


def normalise_yfinance_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert a yfinance download into a flat ``Open/High/Low/Close/Volume`` frame.

    yfinance returns MultiIndex columns ``(Price, Ticker)`` by default in recent versions, and
    flat columns in older ones. Both are handled explicitly.

    Args:
        raw: Frame returned by ``yf.download`` for a single ticker.
        ticker: The ticker that was requested (used to pick the column level).

    Returns:
        A new frame indexed by a tz-naive ``DatetimeIndex`` named ``Date`` with float columns
        ``PRICE_COLUMNS``, sorted ascending, rows with all-NaN prices removed.

    Raises:
        ValueError: If any required price column is missing.
    """
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        level_with_ticker = next(
            (i for i in range(df.columns.nlevels) if ticker in df.columns.get_level_values(i)),
            None,
        )
        if level_with_ticker is not None:
            df = pd.DataFrame(df.xs(ticker, axis=1, level=level_with_ticker))
        else:
            df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}")
    df = df.loc[:, list(PRICE_COLUMNS)].astype(float)
    index = pd.DatetimeIndex(pd.to_datetime(df.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    df.index = index.normalize()
    df.index.name = "Date"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(how="all", subset=["Open", "High", "Low", "Close"])


def download_prices(ticker: str, config: DataConfig | None = None) -> pd.DataFrame:
    """Download daily prices for one ticker from Yahoo Finance.

    Args:
        ticker: Yahoo Finance symbol, e.g. ``"MSFT"`` or ``"^GSPC"``.
        config: Date range and adjustment settings.

    Returns:
        Normalised OHLCV frame (see :func:`normalise_yfinance_frame`).

    Raises:
        ValueError: If no rows are returned.
    """
    cfg = config or DataConfig()
    logger.info("Downloading %s from %s to %s", ticker, cfg.start, cfg.end)
    raw = yf.download(
        ticker,
        start=cfg.start,
        end=cfg.end,
        auto_adjust=cfg.auto_adjust,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"No price data returned for {ticker}")
    return normalise_yfinance_frame(raw, ticker)


def cache_path(ticker: str, data_dir: Path) -> Path:
    """Return the Parquet cache file for ``ticker`` under ``data_dir``."""
    return Path(data_dir) / "prices" / f"{_SAFE_NAME.sub('_', ticker.upper())}.parquet"


def save_prices(prices: pd.DataFrame, ticker: str, data_dir: Path) -> Path:
    """Write a price frame to the Parquet cache and return its path."""
    path = cache_path(ticker, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)
    logger.info("Cached %d rows for %s at %s", len(prices), ticker, path)
    return path


def load_prices(
    ticker: str,
    config: DataConfig | None = None,
    *,
    allow_download: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load prices from the Parquet cache, optionally downloading on a cache miss.

    Web requests call this with ``allow_download=False`` so they never touch the network.

    Args:
        ticker: Yahoo Finance symbol.
        config: Data settings (cache directory, date range).
        allow_download: Download and cache the ticker if it is not cached yet.
        refresh: Force a fresh download even if a cache file exists.

    Returns:
        Normalised OHLCV frame.

    Raises:
        FileNotFoundError: If the ticker is not cached and downloading is not allowed.
    """
    cfg = config or DataConfig()
    path = cache_path(ticker, cfg.data_dir)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    if not allow_download:
        raise FileNotFoundError(f"No cached prices for {ticker} at {path}")
    prices = download_prices(ticker, cfg)
    save_prices(prices, ticker, cfg.data_dir)
    return prices


def list_cached_tickers(data_dir: Path) -> list[str]:
    """Return tickers with a Parquet cache file, sorted alphabetically."""
    folder = Path(data_dir) / "prices"
    if not folder.exists():
        return []
    return sorted(p.stem for p in folder.glob("*.parquet"))
