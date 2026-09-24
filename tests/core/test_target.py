from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockml.config import TargetConfig
from stockml.features.target import (
    make_price_rise_target,
    make_target,
    make_volatility_target,
    typical_absolute_return,
    volatility_persistence_score,
)
from tests.conftest import make_prices


def test_target_compares_next_close() -> None:
    close = pd.Series([1.0, 2.0, 1.5, 1.5, 3.0])
    y = make_price_rise_target(close)
    assert y.iloc[:4].tolist() == [1, 0, 0, 1]
    assert pd.isna(y.iloc[-1])
    assert y.name == "price_rise"


def test_volatility_target_hand_computed() -> None:
    # Absolute log moves: -, 0.1, 0.3, 0.2, 0.05, 0.4
    close = pd.Series(np.exp(np.cumsum([0.0, 0.1, -0.3, 0.2, 0.05, -0.4])))
    threshold = typical_absolute_return(close, 3)
    assert threshold.iloc[:3].isna().all()
    assert threshold.iloc[3] == pytest.approx(0.2)  # median(0.1, 0.3, 0.2)
    assert threshold.iloc[4] == pytest.approx(0.2)  # median(0.3, 0.2, 0.05)
    y = make_volatility_target(close, 3)
    assert y.name == "big_move"
    assert y.iloc[:3].isna().all()  # threshold warm-up
    assert y.iloc[3] == 0  # next move 0.05 <= 0.2
    assert y.iloc[4] == 1  # next move 0.4 > 0.2
    assert pd.isna(y.iloc[5])  # no next day


def test_make_target_dispatches_on_kind() -> None:
    close = make_prices(300, seed=1)["Close"]
    assert make_target(close).name == "price_rise"
    assert make_target(close, TargetConfig(kind="volatility")).name == "big_move"


def test_volatility_target_and_rule_do_not_look_further_ahead() -> None:
    base = make_prices(400, seed=8)["Close"]
    altered = base.copy()
    altered.iloc[300:] *= np.exp(np.linspace(0.0, 0.5, 100))  # change every move from day 300
    cfg = TargetConfig(kind="volatility", volatility_window=60)
    ya, yb = make_target(base, cfg), make_target(altered, cfg)
    # Row t looks at the move into t+1, so rows up to 298 are unaffected.
    pd.testing.assert_series_equal(ya.iloc[:299], yb.iloc[:299])
    sa = volatility_persistence_score(base, cfg)
    sb = volatility_persistence_score(altered, cfg)
    pd.testing.assert_series_equal(sa.iloc[:300], sb.iloc[:300])


def test_persistence_score_is_recent_over_typical() -> None:
    close = make_prices(300, seed=2)["Close"]
    cfg = TargetConfig(kind="volatility", volatility_window=50, persistence_window=5)
    score = volatility_persistence_score(close, cfg)
    moves = np.log(close / close.shift(1)).abs()
    t = 200
    expected = moves.iloc[t - 4 : t + 1].mean() / moves.iloc[t - 49 : t + 1].median()
    assert score.iloc[t] == pytest.approx(expected)
