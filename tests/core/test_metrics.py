from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockml.evaluation.metrics import (
    always_up_metrics,
    annualised_return,
    calmar,
    classification_metrics,
    drawdown,
    equity_curve,
    evaluate_model,
    max_drawdown,
    rolling_sharpe,
    sharpe,
    sortino,
)
from stockml.features.pipeline import build_feature_frame
from stockml.models.registry import build_pipeline
from tests.conftest import make_prices

R = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])


def test_sharpe_annualised_with_sqrt_252() -> None:
    expected = R.mean() / R.std(ddof=1) * np.sqrt(252)
    assert sharpe(R) == pytest.approx(expected)
    assert sharpe(pd.Series([0.01, 0.01, 0.01])) == 0.0


def test_sortino_uses_downside_deviation() -> None:
    downside = np.sqrt(np.mean(np.minimum(R, 0) ** 2))
    assert sortino(R) == pytest.approx(R.mean() / downside * np.sqrt(252))


def test_drawdown_and_max_drawdown() -> None:
    equity = pd.Series([1.0, 1.2, 0.9, 1.1, 1.3])
    assert drawdown(equity).min() == pytest.approx(0.9 / 1.2 - 1)
    assert max_drawdown(equity) == pytest.approx(0.25)


def test_equity_curve_from_log_returns() -> None:
    eq = equity_curve(pd.Series([np.log(1.1), np.log(0.5)]))
    assert eq.tolist() == pytest.approx([1.1, 0.55])


def test_calmar_is_annual_return_over_max_drawdown() -> None:
    mdd = max_drawdown(equity_curve(R))
    assert calmar(R) == pytest.approx(annualised_return(R) / mdd)


def test_rolling_sharpe_window() -> None:
    out = rolling_sharpe(R, 3)
    assert out.isna().sum() == 2


def test_classification_metrics_and_baseline() -> None:
    y = pd.Series([1, 0, 1, 1])
    m = classification_metrics(y, pd.Series([1, 0, 0, 1]), pd.Series([0.9, 0.1, 0.4, 0.8]))
    assert m["accuracy"] == 0.75
    assert m["precision"] == 1.0
    assert m["roc_auc"] == 1.0
    base = always_up_metrics(y)
    assert base["accuracy"] == 0.75
    assert base["recall"] == 1.0
    assert "mse" not in m and "r2" not in m


def test_evaluate_model_is_consistent() -> None:
    X, y = build_feature_frame(make_prices(300, seed=4))
    pipe = build_pipeline("linear_svc").fit(X.iloc[:200], y.iloc[:200])
    ev = evaluate_model(pipe, X.iloc[200:], y.iloc[200:])
    assert (ev.predictions.to_numpy() == pipe.predict(X.iloc[200:])).all()
    assert sum(map(sum, ev.confusion)) == len(X) - 200
    assert ev.metrics["roc_auc"] == pytest.approx(ev.roc.auc)
