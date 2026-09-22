from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockml.config import DataConfig
from stockml.data.cleaning import clean_prices, detect_outliers, replace_outliers, validate_prices
from tests.conftest import make_prices


def test_validate_drops_bad_rows_and_sorts() -> None:
    df = make_prices(10)
    df.iloc[3, df.columns.get_loc("Close")] = np.nan
    df.iloc[4, df.columns.get_loc("Open")] = -1.0
    shuffled = df.iloc[::-1]
    out = validate_prices(shuffled)
    assert len(out) == 8
    assert out.index.is_monotonic_increasing
    assert (out["High"] >= out[["Open", "Close"]].max(axis=1)).all()


def test_validate_missing_column_raises() -> None:
    with pytest.raises(ValueError):
        validate_prices(make_prices(5).drop(columns="High"))


def test_validate_does_not_mutate_input() -> None:
    df = make_prices(10)
    before = df.copy()
    validate_prices(df)
    pd.testing.assert_frame_equal(df, before)


def _spike_series() -> pd.Series:
    close = make_prices(200, seed=1)["Close"].copy()
    close.iloc[100] = close.iloc[99] * 1.8  # one-day bad tick that snaps back
    return close


@pytest.mark.parametrize("method", ["zscore", "iqr"])
def test_detect_outliers_flags_spike(method: str) -> None:
    mask = detect_outliers(_spike_series(), DataConfig(outlier_method=method))  # type: ignore[arg-type]
    assert list(mask[mask].index) == [_spike_series().index[100]]


def test_detect_outliers_ignores_genuine_crash() -> None:
    close = make_prices(200, seed=1)["Close"].copy()
    close.iloc[100:] = close.iloc[100:] * 0.5  # permanent level shift, no reversal
    assert not detect_outliers(close).any()


def test_replace_outliers_uses_previous_close() -> None:
    df = make_prices(200, seed=1)
    df.iloc[100, df.columns.get_loc("Close")] *= 1.8
    mask = detect_outliers(df["Close"])
    out = replace_outliers(df, mask)
    assert out["Close"].iloc[100] == pytest.approx(df["Close"].iloc[99])
    assert df["Close"].iloc[100] != out["Close"].iloc[100]  # input untouched


def test_clean_prices_runs_end_to_end() -> None:
    out = clean_prices(make_prices(50))
    assert len(out) == 50
