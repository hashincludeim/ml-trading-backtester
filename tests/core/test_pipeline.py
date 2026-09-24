from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockml.config import FeatureConfig, TargetConfig
from stockml.features.pipeline import build_feature_frame, compute_features, compute_indicators
from tests.conftest import make_prices


def test_feature_frame_columns_and_alignment(prices: pd.DataFrame) -> None:
    cfg = FeatureConfig()
    X, y = build_feature_frame(prices, cfg)
    assert tuple(X.columns) == cfg.feature_columns
    assert X.index.equals(y.index)
    assert not X.isna().any().any()
    assert set(y.unique()) <= {0, 1}
    assert X.index[-1] < prices.index[-1]  # final (unlabelled) row dropped


def test_ema_bullish_is_one_when_short_above_long(prices: pd.DataFrame) -> None:
    cfg = FeatureConfig()
    X, _ = build_feature_frame(prices, cfg)
    ind = compute_indicators(prices, cfg).loc[X.index]
    short, long = ind[f"ema_{cfg.ema_cross_short}"], ind[f"ema_{cfg.ema_cross_long}"]
    assert (X["ema_bullish"] == (short > long).astype(float)).all()


def test_features_do_not_use_future_prices() -> None:
    base = make_prices(250, seed=7)
    altered = base.copy()
    altered.iloc[180:] *= 1.5
    altered.iloc[180:, altered.columns.get_loc("Volume")] *= 3
    fa, fb = compute_features(base), compute_features(altered)
    pd.testing.assert_frame_equal(fa.iloc[:180], fb.iloc[:180])


def test_target_is_only_thing_that_sees_next_day() -> None:
    base = make_prices(250, seed=7)
    altered = base.copy()
    altered.iloc[200, altered.columns.get_loc("Close")] *= 2
    Xa, ya = build_feature_frame(base)
    Xb, yb = build_feature_frame(altered)
    pd.testing.assert_frame_equal(Xa.loc[: base.index[199]], Xb.loc[: base.index[199]])
    assert ya.loc[base.index[199]] != yb.loc[base.index[199]] or ya.loc[base.index[199]] == 1


def test_compute_features_does_not_mutate(prices: pd.DataFrame) -> None:
    before = prices.copy()
    compute_features(prices)
    pd.testing.assert_frame_equal(prices, before)


def test_volatility_target_drops_its_warm_up() -> None:
    prices = make_prices(400, seed=9)
    target = TargetConfig(kind="volatility", volatility_window=100)
    X, y = build_feature_frame(prices, target=target)
    assert X.index.equals(y.index)
    assert X.index[0] >= prices.index[100]
    assert set(y.unique()) == {0, 1}
    X_dir, _ = build_feature_frame(prices)
    pd.testing.assert_frame_equal(X, X_dir.loc[X.index])  # same features, different question


def test_longer_volatility_features() -> None:
    prices = make_prices(300, seed=4)
    cfg = FeatureConfig()
    assert cfg.volatility_ratio_column == "vol_ratio_5_63"
    X, _ = build_feature_frame(prices, cfg)
    returns = np.log(prices["Close"] / prices["Close"].shift(1))
    t = X.index[-1]
    window_63 = returns.loc[:t].iloc[-63:]
    assert X.loc[t, "volatility_63"] == pytest.approx(window_63.std())
    assert X.loc[t, "volatility_21"] == pytest.approx(window_63.iloc[-21:].std())
    ratio = X["volatility_5"] / X["volatility_63"]
    pd.testing.assert_series_equal(X["vol_ratio_5_63"], ratio, check_names=False)
    assert X.index[0] >= prices.index[63]  # the longest window sets the warm-up
