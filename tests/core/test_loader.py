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
    save_prices,
)
from tests.conftest import make_prices


def _multiindex_frame(ticker: str) -> pd.DataFrame:
    flat = make_prices(5)
    flat.columns = pd.MultiIndex.from_product([flat.columns, [ticker]], names=["Price", "Ticker"])
    return flat


def test_normalise_handles_multiindex_columns() -> None:
    out = normalise_yfinance_frame(_multiindex_frame("AMZN"), "AMZN")
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
        load_prices("AMZN", DataConfig(data_dir=tmp_path))


def test_load_prices_downloads_caches_and_reads_back(tmp_path: Path) -> None:
    cfg = DataConfig(data_dir=tmp_path)
    with patch("stockml.data.loader.yf.download", return_value=_multiindex_frame("AMZN")) as dl:
        first = load_prices("AMZN", cfg, allow_download=True)
        second = load_prices("AMZN", cfg, allow_download=True)
    assert dl.call_count == 1
    assert dl.call_args.kwargs["auto_adjust"] is True
    pd.testing.assert_frame_equal(first, second, check_freq=False)
    assert cache_path("AMZN", tmp_path).exists()
    assert list_cached_tickers(tmp_path) == ["AMZN"]


def test_download_empty_raises(tmp_path: Path) -> None:
    with (
        patch("stockml.data.loader.yf.download", return_value=pd.DataFrame()),
        pytest.raises(ValueError, match="No price data"),
    ):
        load_prices("NOPE", DataConfig(data_dir=tmp_path), allow_download=True)


def test_index_symbol_round_trips_through_cache(tmp_path: Path) -> None:
    save_prices(make_prices(5), "^GSPC", tmp_path)
    assert cache_path("^GSPC", tmp_path).name == "^GSPC.parquet"
    assert list_cached_tickers(tmp_path) == ["^GSPC"]
    assert len(load_prices("^GSPC", DataConfig(data_dir=tmp_path))) == 5
