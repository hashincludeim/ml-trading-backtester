from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from stockml.config import DataConfig
from stockml.data.loader import (
    cache_path,
    list_cached_tickers,
    load_prices,
    normalise_yfinance_frame,
)
from tests.conftest import make_prices


def _multiindex_frame(ticker: str) -> pd.DataFrame:
    flat = make_prices(5)
    flat.columns = pd.MultiIndex.from_product([flat.columns, [ticker]], names=["Price", "Ticker"])
    return flat


def test_normalise_handles_multiindex_columns() -> None:
    out = normalise_yfinance_frame(_multiindex_frame("BARC.L"), "BARC.L")
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out.index.name == "Date"


def test_normalise_handles_flat_lowercase_and_tz() -> None:
    raw = make_prices(5)
    raw.columns = [c.lower() for c in raw.columns]
    raw.index = raw.index.tz_localize("Europe/London")
    out = normalise_yfinance_frame(raw, "X")
    assert out.index.tz is None
    assert "Close" in out.columns


def test_normalise_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        normalise_yfinance_frame(make_prices(5).drop(columns="Volume"), "X")


def test_load_prices_without_cache_and_no_download_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_prices("BARC.L", DataConfig(data_dir=tmp_path))


def test_load_prices_downloads_caches_and_reads_back(tmp_path: Path) -> None:
    cfg = DataConfig(data_dir=tmp_path)
    with patch("stockml.data.loader.yf.download", return_value=_multiindex_frame("BARC.L")) as dl:
        first = load_prices("BARC.L", cfg, allow_download=True)
        second = load_prices("BARC.L", cfg, allow_download=True)
    assert dl.call_count == 1
    assert dl.call_args.kwargs["auto_adjust"] is True
    pd.testing.assert_frame_equal(first, second, check_freq=False)
    assert cache_path("BARC.L", tmp_path).exists()
    assert list_cached_tickers(tmp_path) == ["BARC.L"]


def test_download_empty_raises(tmp_path: Path) -> None:
    with (
        patch("stockml.data.loader.yf.download", return_value=pd.DataFrame()),
        pytest.raises(ValueError, match="No price data"),
    ):
        load_prices("NOPE", DataConfig(data_dir=tmp_path), allow_download=True)
