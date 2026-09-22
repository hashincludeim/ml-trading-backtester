from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from stockml.evaluation.metrics import RocCurve
from stockml.features.pipeline import build_feature_frame, compute_indicators
from stockml.viz import charts
from stockml.viz.theme import MODEL_COLORS, TEMPLATE_NAME
from tests.conftest import make_prices


def _title(fig: go.Figure) -> str:
    return str(fig.layout.title.text)


def _assert_themed(fig: go.Figure) -> None:
    assert isinstance(fig, go.Figure)
    assert _title(fig)
    assert (
        fig.layout.template.layout.paper_bgcolor is not None or fig.layout.template == TEMPLATE_NAME
    )


def test_price_volume_chart_downsamples() -> None:
    prices = make_prices(3_000)
    fig = charts.price_volume_chart(prices, "TEST", max_points=1_000)
    _assert_themed(fig)
    assert [t.type for t in fig.data] == ["candlestick", "bar"]
    assert len(fig.data[0].x) <= 1_000
    assert "weekly" in _title(fig)


def test_indicator_chart_has_three_panels() -> None:
    fig = charts.indicator_chart(compute_indicators(make_prices(200)), "TEST")
    names = {t.name for t in fig.data}
    assert {"Close", "SMA 10", "EMA 30", "RSI", "MACD line"} <= names
    assert fig.layout.yaxis3 is not None
    assert _title(fig) == "TEST technical indicators"


def test_exploration_charts() -> None:
    X, y = build_feature_frame(make_prices(300))
    dist = charts.feature_distribution_chart(X, y)
    assert len(dist.data) == 2 * X.shape[1]
    assert len(dist.layout.updatemenus[0].buttons) == X.shape[1]
    heat = charts.correlation_heatmap(X)
    assert heat.data[0].type == "heatmap"
    bal = charts.target_balance_chart(y)
    assert bal.data[0].type == "bar"
    for fig in (dist, heat, bal):
        _assert_themed(fig)


def test_returns_histogram_is_daily() -> None:
    returns = pd.DataFrame({"random_forest": [0.01, -0.02], "Buy & hold": [0.0, 0.01]})
    fig = charts.returns_histogram(returns, "Daily returns")
    assert list(fig.data[0].x) == pytest.approx([1.0, -2.0])
    assert fig.data[0].marker.color == MODEL_COLORS["random_forest"]
    assert fig.layout.xaxis.title.text == "Daily log return (%)"


def test_model_charts_use_consistent_colours() -> None:
    roc = RocCurve([0, 0.5, 1], [0, 0.6, 1], 0.55)
    fig = charts.roc_curves_chart({"svm_rbf": roc, "extra_trees": roc})
    assert fig.data[1].line.color == MODEL_COLORS["svm_rbf"]
    assert fig.data[2].line.color == MODEL_COLORS["extra_trees"]
    cv = charts.cv_scores_chart({"svm_rbf": (0.51, 0.01)}, baseline=0.5)
    assert cv.data[0].marker.color[0] == MODEL_COLORS["svm_rbf"]
    cm = charts.confusion_matrix_chart([[10, 5], [3, 12]], "svm_rbf")
    assert "SVM (RBF)" in _title(cm)
    imp = pd.DataFrame(
        {"importance_mean": [0.02, 0.01], "importance_std": [0.0, 0.0]}, index=["a", "b"]
    )
    fi = charts.feature_importance_chart(imp, "svm_rbf")
    assert list(fi.data[0].y) == ["b", "a"]


def test_backtest_charts() -> None:
    idx = pd.bdate_range("2024-01-01", periods=5)
    frame = pd.DataFrame(
        {"svm_rbf": [1, 1.1, 1.2, 1.1, 1.3], "Buy & hold": [1, 1, 1, 1, 1.1]}, index=idx
    )
    eq = charts.equity_curves_chart(frame, 5)
    assert eq.data[1].line.dash == "dash"
    assert eq.layout.yaxis.type == "log"
    assert charts.drawdown_chart(frame - 1).data[0].type in {"scatter", "scattergl"}
    assert "126" in _title(charts.rolling_sharpe_chart(frame, 126))


def test_multi_ticker_charts() -> None:
    matrix = pd.DataFrame({"svm_rbf": [0.51, 0.49]}, index=["A.L", "B.L"])
    fig = charts.metric_heatmap(matrix, "Accuracy", center=0.5)
    assert fig.data[0].zmid is None and fig.data[0].zmin == pytest.approx(0.49)
    bars = pd.DataFrame(
        {"model": ["svm_rbf", "svm_rbf"], "strategy": [0.3, -0.1], "buy_and_hold": [0.2, 0.1]},
        index=["A.L", "B.L"],
    )
    assert len(charts.strategy_vs_benchmark_chart(bars, "Sharpe").data) == 2
    closes = pd.DataFrame(
        {"A.L": [1.0, 2.0], "B.L": [2.0, 2.0]}, index=pd.bdate_range("2024-01-01", periods=2)
    )
    norm = charts.normalised_prices_chart(closes)
    assert list(norm.data[0].y) == [100.0, 200.0]
