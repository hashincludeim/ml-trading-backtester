from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from stockml import analysis
from stockml.config import TARGET_LABELS
from stockml.evaluation.metrics import RocCurve
from stockml.features.pipeline import build_feature_frame, compute_indicators
from stockml.viz import charts
from stockml.viz.theme import (
    AXIS_LINE,
    BENCHMARK_FILL,
    BIG_MOVE_COLOR,
    CATEGORICAL,
    CATEGORICAL_DARK,
    CONTROL_ACTIVE,
    CONTROL_BG,
    DARK_COLOR_MAP,
    DIVERGING_MIDPOINT,
    FONT_FAMILY,
    GRID,
    MODEL_COLORS,
    SURFACE,
    SURFACE_RAISED,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from tests.conftest import make_prices


def _title(fig: go.Figure) -> str:
    return str(fig.layout.title.text)


def _assert_themed(fig: go.Figure) -> None:
    assert isinstance(fig, go.Figure)
    assert _title(fig)
    assert fig.layout.template.layout.font.family == FONT_FAMILY
    assert fig.layout.template.layout.paper_bgcolor == SURFACE


def test_dark_map_covers_palette_one_to_one() -> None:
    for light, dark in zip(CATEGORICAL, CATEGORICAL_DARK, strict=True):
        assert DARK_COLOR_MAP[light] == dark
    assert DARK_COLOR_MAP[SURFACE] != SURFACE


def test_chrome_colours_are_distinct_and_all_have_dark_steps() -> None:
    # The browser swaps colours by exact string, so two roles sharing a light hex would be
    # forced onto one dark colour (e.g. hover labels vanishing into the chart surface).
    chrome = [
        SURFACE,
        SURFACE_RAISED,
        TEXT_PRIMARY,
        TEXT_SECONDARY,
        TEXT_MUTED,
        GRID,
        AXIS_LINE,
        CONTROL_BG,
        CONTROL_ACTIVE,
        BENCHMARK_FILL,
        DIVERGING_MIDPOINT,
    ]
    assert len({c.lower() for c in chrome}) == len(chrome)
    for colour in chrome:
        assert DARK_COLOR_MAP[colour] != colour


def test_price_volume_chart_downsamples_with_overlays_and_events() -> None:
    prices = make_prices(3_000, start="2005-01-03")
    overlays = pd.DataFrame({"SMA 50": prices["Close"].rolling(50).mean()})
    fig = charts.price_volume_chart(
        prices, "TEST", overlays, events=(("2008-09-15", "Lehman"), ("1990-01-01", "Too old"))
    )
    _assert_themed(fig)
    assert [t.type for t in fig.data][:1] == ["candlestick"]
    assert len(fig.data[0].x) <= 1_500
    assert fig.data[0].name == "Weekly OHLC"
    assert "SMA 50" in {t.name for t in fig.data}
    labels = [a.text for a in fig.layout.annotations]
    assert "Lehman" in labels and "Too old" not in labels
    events = [a for a in fig.layout.annotations if a.name == charts.EVENT_ANNOTATION]
    assert [a.text for a in events] == ["Lehman"]
    # Event labels near the right edge must not stretch the time axis past the data.
    limits = fig.layout.xaxis.autorangeoptions
    # Epoch milliseconds: plotly.js ignores date strings here and falls back to 2000-2001.
    assert isinstance(limits.minallowed, int) and isinstance(limits.maxallowed, int)
    assert prices.index[0] <= pd.Timestamp(limits.minallowed, unit="ms")
    assert pd.Timestamp(limits.maxallowed, unit="ms") <= prices.index[-1]


def test_indicator_chart_has_three_panels() -> None:
    fig = charts.indicator_chart(compute_indicators(make_prices(200)), "TEST")
    names = {t.name for t in fig.data}
    assert {"Close", "SMA 10", "EMA 30", "RSI", "MACD", "Signal"} <= names
    assert fig.layout.yaxis3 is not None
    assert _title(fig) == "TEST technical indicators"
    assert fig.layout.title.subtitle.text


def test_history_charts() -> None:
    close = make_prices(600)["Close"]
    rets = np.log(close).diff()
    ep = analysis.worst_drawdown(close)
    under = charts.underwater_chart(close, "TEST", ep)
    assert under.data[0].fill == "tozeroy"
    assert f"{ep.depth:.0%}" in under.layout.annotations[0].text
    annual = charts.annual_returns_chart(analysis.annual_returns(rets), "Years")
    assert annual.data[0].type == "bar"
    table = analysis.calendar_returns(rets)
    heat = charts.monthly_returns_heatmap(table, "Months")
    assert list(heat.data[0].y)[-1] == "Avg"
    vol = charts.volatility_chart(analysis.rolling_annualised_volatility(close, 20), "TEST", 20)
    assert "20-day" in _title(vol)
    for fig in (under, annual, heat, vol):
        _assert_themed(fig)


def test_exploration_charts() -> None:
    X, y = build_feature_frame(make_prices(300))
    dist = charts.feature_distribution_chart(X, y)
    assert len(dist.data) == 2 * X.shape[1]
    assert len(dist.layout.updatemenus[0].buttons) == X.shape[1]
    assert dist.data[0].line.shape == "hvh"
    stats = {f: analysis.feature_bucket_stats(X[f], y, 5) for f in X.columns}
    signal = charts.feature_signal_chart(stats, float(y.mean()), 5)
    assert len(signal.data) == X.shape[1]
    assert sum(bool(t.visible) for t in signal.data) == 1
    corr = pd.Series({"a": 0.05, "b": -0.01})
    fc = charts.feature_correlation_chart(corr, 0.03)
    assert list(fc.data[0].y) == ["b", "a"]
    acf = charts.autocorrelation_chart(pd.Series([0.1, 0.0], index=[1, 2]), 0.05, "TEST")
    assert acf.data[0].marker.color[0] != acf.data[0].marker.color[1]
    heat = charts.correlation_heatmap(X)
    assert heat.data[0].type == "heatmap"
    bal = charts.target_balance_chart(y)
    assert bal.data[0].type == "bar"
    for fig in (dist, signal, fc, acf, heat, bal):
        _assert_themed(fig)


def test_indicator_signal_chart() -> None:
    table = analysis.indicator_signal_table(compute_indicators(make_prices(400)))
    fig = charts.indicator_signal_chart(table, 0.5)
    assert {t.name for t in fig.data} == set(table["indicator"])
    _assert_themed(fig)


def test_returns_histogram_is_daily_and_fits_normal() -> None:
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        {"random_forest": rng.normal(0, 0.01, 500), "Buy & hold": rng.normal(0, 0.01, 500)}
    )
    fig = charts.returns_histogram(returns, "Daily returns", bins=40, fit_normal=True)
    assert fig.data[0].marker.color == MODEL_COLORS["random_forest"]
    assert sum(fig.data[0].y) == 500  # every day counted once (tails folded in)
    assert max(abs(x) for x in fig.data[0].x) < 5  # percent, daily scale
    assert fig.data[-1].name.startswith("Normal")
    assert fig.layout.xaxis.title.text == "Daily log return"


def test_model_charts_use_consistent_colours() -> None:
    roc = RocCurve([0, 0.5, 1], [0, 0.6, 1], 0.55)
    fig = charts.roc_curves_chart({"svm_rbf": roc, "extra_trees": roc})
    assert fig.data[1].line.color == MODEL_COLORS["svm_rbf"]
    assert fig.data[2].line.color == MODEL_COLORS["extra_trees"]
    cv = charts.cv_scores_chart({"svm_rbf": (0.51, 0.01)}, baseline=0.5)
    assert cv.data[0].marker.color[0] == MODEL_COLORS["svm_rbf"]
    folds = charts.cv_folds_chart({"svm_rbf": [0.49, 0.52, 0.5]})
    assert list(folds.data[0].x) == ["Fold 1", "Fold 2", "Fold 3"]
    cm = charts.confusion_matrix_chart([[10, 5], [3, 12]], "svm_rbf")
    assert "SVM (RBF)" in _title(cm)
    assert cm.data[0].z[0][0] == pytest.approx(10 / 15)
    imp = pd.DataFrame(
        {"importance_mean": [0.02, -0.01], "importance_std": [0.0, 0.0]}, index=["a", "b"]
    )
    fi = charts.feature_importance_chart(imp, "svm_rbf")
    assert list(fi.data[0].y) == ["b", "a"]
    scores = pd.Series(np.linspace(-1, 1, 100))
    truth = pd.Series([0, 1] * 50)
    sd = charts.score_distribution_chart(scores, truth, "svm_rbf")
    assert len(sd.data) == 2
    idx = pd.bdate_range("2024-01-01", periods=5)
    ra = charts.rolling_accuracy_chart(pd.DataFrame({"svm_rbf": [0.5] * 5}, index=idx), 63)
    assert ra.data[0].line.color == MODEL_COLORS["svm_rbf"]


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
    costs = pd.DataFrame({"svm_rbf": [1.0, 0.5], "buy_and_hold": [0.2, 0.2]}, index=[0.0, 10.0])
    cs = charts.cost_sensitivity_chart(costs, 5)
    assert [t.name for t in cs.data] == ["SVM (RBF)", "Buy & hold"]
    points = pd.DataFrame(
        {
            "annual_volatility": [0.2, 0.3],
            "annual_return": [0.05, -0.02],
            "sharpe": [0.25, -0.07],
            "color": [MODEL_COLORS["svm_rbf"], "#000000"],
            "benchmark": [False, True],
        },
        index=["SVM (RBF)", "Buy & hold"],
    )
    rr = charts.risk_return_scatter(points, "Risk", "hint")
    assert rr.data[1].marker.symbol == "diamond"


def test_multi_ticker_charts() -> None:
    matrix = pd.DataFrame({"svm_rbf": [0.51, 0.49]}, index=["A.L", "B.L"])
    fig = charts.metric_heatmap(matrix, "Accuracy", center=0.5)
    assert fig.data[0].zmin == pytest.approx(0.49)
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


def test_indicator_chart_shades_rsi_zones() -> None:
    fig = charts.indicator_chart(compute_indicators(make_prices(200)), "TEST")
    rects = [s for s in fig.layout.shapes if s.type == "rect"]
    assert len(rects) == 2
    assert all(s.yref == "y2" for s in rects)


def test_reference_lines_add_no_placeholder_annotations() -> None:
    closes = pd.DataFrame({"A.L": [1.0, 2.0, 4.0]}, index=pd.bdate_range("2024-01-01", periods=3))
    fig = charts.normalised_prices_chart(closes)
    assert all(a.text != "new text" for a in fig.layout.annotations)


def test_equity_break_even_label_sits_at_one_pound_on_log_axis() -> None:
    idx = pd.bdate_range("2024-01-01", periods=3)
    fig = charts.equity_curves_chart(pd.DataFrame({"svm_rbf": [1, 2, 4.0]}, index=idx), 0)
    label = next(a for a in fig.layout.annotations if a.text.startswith("Break-even"))
    assert label.y == 0  # log10(1)


def test_walk_forward_chart_pairs_static_and_walk_forward_intervals() -> None:
    frame = pd.DataFrame(
        {
            "static_auc": [0.52, 0.49],
            "static_low": [0.49, 0.46],
            "static_high": [0.55, 0.52],
            "wf_auc": [0.53, 0.50],
            "wf_low": [0.50, 0.47],
            "wf_high": [0.56, 0.53],
        },
        index=["svm_rbf", "logistic_regression"],
    )
    fig = charts.walk_forward_chart(frame, 63)
    _assert_themed(fig)
    assert [t.name for t in fig.data] == [charts.TRAINED_ONCE_LABEL, charts.WALK_FORWARD_LABEL]
    wf = fig.data[1]
    assert list(wf.x) == [0.53, 0.50]
    assert list(wf.error_x.array) == pytest.approx([0.03, 0.03])
    assert list(wf.error_x.arrayminus) == pytest.approx([0.03, 0.03])
    assert list(fig.layout.yaxis.ticktext) == ["SVM (RBF)", "Logistic Regression"]
    assert fig.layout.yaxis.range[0] > fig.layout.yaxis.range[1]  # first model on top
    lo, hi = fig.layout.xaxis.range
    assert lo < 0.46 and hi > 0.56  # intervals and the 0.5 line stay in view
    assert [s.x0 for s in fig.layout.shapes] == [0.5]
    assert not fig.layout.annotations  # no "new text" placeholder from add_vline
    assert "63 trading days" in fig.layout.title.subtitle.text


def test_class_charts_use_target_wording_and_colours() -> None:
    vol = TARGET_LABELS["volatility"]
    cm = charts.confusion_matrix_chart([[10, 5], [3, 12]], "svm_rbf", vol)
    assert list(cm.data[0].y) == ["Actual Quiet", "Actual Big move"]
    assert list(charts.confusion_matrix_chart([[1, 0], [0, 1]], "svm_rbf").data[0].x) == [
        "Predicted Down",
        "Predicted Up",
    ]
    scores = pd.Series(np.linspace(-1, 1, 100))
    truth = pd.Series([0, 1] * 50)
    sd = charts.score_distribution_chart(scores, truth, "svm_rbf", target=vol)
    assert sd.data[0].name.startswith("Actual: big move")
    assert sd.data[0].line.color == BIG_MOVE_COLOR
    assert sd.data[0].fillcolor in DARK_COLOR_MAP and sd.data[1].fillcolor in DARK_COLOR_MAP
