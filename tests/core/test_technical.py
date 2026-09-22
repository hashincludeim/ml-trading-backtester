from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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

S = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_sma_hand_computed() -> None:
    out = sma(S, 3)
    assert out.isna().sum() == 2
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[5] == pytest.approx(5.0)


def test_ema_matches_recursive_formula() -> None:
    out = ema(S, 3)
    alpha = 2 / (3 + 1)
    expected = [S.iloc[0]]
    for v in S.iloc[1:]:
        expected.append(alpha * v + (1 - alpha) * expected[-1])
    assert out.iloc[2:].tolist() == pytest.approx(expected[2:])
    assert out.iloc[:2].isna().all()


def test_log_returns_and_volatility() -> None:
    r = log_returns(pd.Series([100.0, 110.0, 99.0]))
    assert r.iloc[1] == pytest.approx(np.log(1.1))
    vol = rolling_volatility(pd.Series([100.0, 101, 102, 101, 100]), 2)
    assert vol.notna().sum() == 3


def test_rsi_bounds_and_extremes() -> None:
    up = pd.Series(np.arange(1, 40, dtype=float))
    assert rsi(up, 14).dropna().eq(100.0).all()
    down = up[::-1].reset_index(drop=True)
    assert rsi(down, 14).dropna().iloc[-1] == pytest.approx(0.0)
    noisy = pd.Series(100 + np.random.default_rng(0).normal(0, 1, 200).cumsum())
    vals = rsi(noisy).dropna()
    assert ((vals >= 0) & (vals <= 100)).all()


def test_macd_uses_its_argument() -> None:
    a = pd.Series(np.linspace(1, 50, 60))
    b = a * 2
    ma, mb = macd(a), macd(b)
    assert list(ma.columns) == ["macd", "signal", "hist"]
    np.testing.assert_allclose(mb["macd"].dropna(), 2 * ma["macd"].dropna())


def test_bollinger_position() -> None:
    close = pd.Series([10.0, 12.0, 10.0, 12.0, 14.0])
    bb = bollinger(close, window=3, num_std=2.0)
    row = bb.iloc[4]
    assert row["upper"] > row["middle"] > row["lower"]
    expected = (14.0 - row["middle"]) / (row["upper"] - row["lower"])
    assert row["position"] == pytest.approx(expected)


def test_ema_crossovers_direction() -> None:
    short = pd.Series([1.0, 1.0, 3.0, 3.0, 1.0])
    long = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])
    cross = ema_crossovers(short, long)
    assert cross["bullish"].tolist() == [False, False, True, False, False]
    assert cross["bearish"].tolist() == [False, False, False, False, True]


@pytest.mark.parametrize(
    "fn",
    [
        lambda s: sma(s, 5),
        lambda s: ema(s, 5),
        lambda s: rsi(s, 5),
        lambda s: macd(s, 3, 6, 3)["hist"],
        lambda s: bollinger(s, 5)["position"],
        lambda s: rolling_volatility(s, 5),
    ],
)
def test_indicators_have_no_look_ahead(fn) -> None:  # type: ignore[no-untyped-def]
    base = pd.Series(100 + np.random.default_rng(3).normal(0, 1, 80).cumsum())
    altered = base.copy()
    altered.iloc[50:] *= 3.0
    pd.testing.assert_series_equal(fn(base).iloc[:50], fn(altered).iloc[:50])
