from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockml import analysis
from stockml.features.pipeline import build_feature_frame, compute_indicators
from tests.conftest import make_prices


def test_calendar_and_annual_returns_compound_log_returns() -> None:
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-02-03", "2021-03-01"])
    r = pd.Series(np.log([1.1, 1.1, 0.5, 2.0]), index=idx)
    table = analysis.calendar_returns(r)
    assert list(table.columns) == list(analysis.MONTH_LABELS)
    assert table.loc[2020, "Jan"] == pytest.approx(1.21 - 1)
    assert table.loc[2020, "Feb"] == pytest.approx(-0.5)
    assert np.isnan(table.loc[2020, "Mar"])
    annual = analysis.annual_returns(r)
    assert annual.loc[2020] == pytest.approx(1.21 * 0.5 - 1)
    assert annual.loc[2021] == pytest.approx(1.0)


def test_worst_drawdown_episode() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    s = pd.Series([100, 120, 60, 90, 130, 125.0], index=idx)
    ep = analysis.worst_drawdown(s)
    assert ep.depth == pytest.approx(0.5)
    assert ep.peak_date == idx[1] and ep.trough_date == idx[2]
    assert ep.recovery_date == idx[4]
    never = analysis.worst_drawdown(pd.Series([100, 50, 60.0], index=idx[:3]))
    assert never.recovery_date is None


def test_rolling_volatility_is_annualised() -> None:
    close = make_prices(100)["Close"]
    vol = analysis.rolling_annualised_volatility(close, 20)
    raw = np.log(close).diff().rolling(20).std()
    assert vol.iloc[-1] == pytest.approx(raw.iloc[-1] * np.sqrt(252))


def test_autocorrelation_and_noise_band() -> None:
    r = pd.Series(np.random.default_rng(0).normal(size=2_000))
    acf = analysis.autocorrelation(r, 5)
    assert list(acf.index) == [1, 2, 3, 4, 5]
    assert (acf.abs() < 0.1).all()
    assert analysis.noise_band(10_000) == pytest.approx(0.0196)


def test_feature_bucket_stats_balanced_bins() -> None:
    x = pd.Series(np.arange(100, dtype=float))
    y = pd.Series((np.arange(100) >= 50).astype(int))
    stats = analysis.feature_bucket_stats(x, y, 4)
    assert len(stats) == 4
    assert stats["n"].tolist() == [25, 25, 25, 25]
    assert stats["up_rate"].tolist() == [0, 0, 1, 1]
    assert (stats["se"] == 0).all()


def test_feature_return_correlation_sorted_by_magnitude() -> None:
    X = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [4.0, 3, 2, 1], "c": [1.0, 3, 2, 4]})
    fwd = pd.Series([1.0, 2, 3, 4])
    corr = analysis.feature_return_correlation(X, fwd)
    assert corr["a"] == pytest.approx(1.0) and corr["b"] == pytest.approx(-1.0)
    assert corr.index[-1] == "c"


def test_indicator_signal_table_states_partition_days() -> None:
    ind = compute_indicators(make_prices(400, seed=3))
    table = analysis.indicator_signal_table(ind)
    assert {"RSI", "Bollinger", "EMA trend", "MACD"} == set(table["indicator"])
    rsi_total = table.loc[table["indicator"] == "RSI", "n"].sum()
    labelled = ind["rsi"].notna() & ind["Close"].shift(-1).notna()
    assert rsi_total == labelled.sum()
    assert table["up_rate"].between(0, 1).all()


def test_rolling_hit_rate() -> None:
    y = pd.Series([1, 0, 1, 1])
    p = pd.Series([1, 1, 1, 0])
    assert analysis.rolling_hit_rate(y, p, 2).tolist()[1:] == [0.5, 0.5, 0.5]


def test_binned_density_integrates_to_one() -> None:
    values = pd.Series(np.random.default_rng(1).normal(size=1_000))
    edges = analysis.shared_edges([values], 30)
    centres, density = analysis.binned_density(values.clip(edges[0], edges[-1]), edges)
    assert len(centres) == 30
    assert float((density * np.diff(edges)).sum()) == pytest.approx(1.0)


def test_analysis_on_feature_frame_runs() -> None:
    X, y = build_feature_frame(make_prices(300))
    stats = analysis.feature_bucket_stats(X["rsi_14"], y, 10)
    assert stats["n"].sum() == len(X)
