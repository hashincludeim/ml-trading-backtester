from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockml.config import BacktestConfig
from stockml.evaluation.backtest import BUY_AND_HOLD, STRATEGY, run_backtest

IDX = pd.bdate_range("2024-01-01", periods=5)
CLOSE = pd.Series([100.0, 110.0, 99.0, 99.0, 108.9], index=IDX)
R = np.log(CLOSE.shift(-1) / CLOSE)  # next-day log returns


def test_long_short_no_costs_hand_computed() -> None:
    preds = pd.Series([1, 0, 1, 1], index=IDX[:4])
    bt = run_backtest(CLOSE, preds, BacktestConfig(cost_bps=0))
    expected = [R.iloc[0], -R.iloc[1], R.iloc[2], R.iloc[3]]
    assert bt.returns[STRATEGY].tolist() == pytest.approx(expected)
    assert bt.returns[BUY_AND_HOLD].tolist() == pytest.approx(R.iloc[:4].tolist())
    assert bt.equity[BUY_AND_HOLD].iloc[-1] == pytest.approx(108.9 / 100.0)


def test_long_flat_holds_cash_on_down_prediction() -> None:
    preds = pd.Series([1, 0, 1, 1], index=IDX[:4])
    bt = run_backtest(CLOSE, preds, BacktestConfig(mode="long_flat", cost_bps=0))
    assert bt.returns[STRATEGY].iloc[1] == 0.0
    assert bt.positions.tolist() == [1.0, 0.0, 1.0, 1.0]


def test_costs_charged_on_position_changes() -> None:
    preds = pd.Series([1, 0, 1, 1], index=IDX[:4])
    bt = run_backtest(CLOSE, preds, BacktestConfig(cost_bps=10))
    c = 10 / 10_000
    # entry (1), flip long->short (2), flip short->long (2), hold (0)
    expected = [R.iloc[0] - c, -R.iloc[1] - 2 * c, R.iloc[2] - 2 * c, R.iloc[3]]
    assert bt.returns[STRATEGY].tolist() == pytest.approx(expected)
    assert bt.metrics[STRATEGY]["n_trades"] == 3
    assert bt.metrics[STRATEGY]["cost_paid"] == pytest.approx(5 * c)
    assert bt.metrics[BUY_AND_HOLD]["n_trades"] == 1


def test_prediction_on_last_day_without_next_close_is_dropped() -> None:
    preds = pd.Series([1, 1, 1, 1, 1], index=IDX)
    bt = run_backtest(CLOSE, preds, BacktestConfig(cost_bps=0))
    assert len(bt.returns) == 4


def test_drawdown_non_positive() -> None:
    preds = pd.Series([1, 1, 1, 1], index=IDX[:4])
    bt = run_backtest(CLOSE, preds)
    assert (bt.drawdown <= 0).all().all()
    assert bt.metrics[STRATEGY]["max_drawdown"] == pytest.approx(-bt.drawdown[STRATEGY].min())
