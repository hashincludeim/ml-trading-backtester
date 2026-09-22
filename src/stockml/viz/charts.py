"""Chart builders. Each function takes data and returns a themed ``plotly.graph_objects.Figure``.

No I/O and no web-framework code lives here, so the same figures render in a notebook, a
static export, or the Django dashboard.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stockml.config import FeatureConfig
from stockml.evaluation.metrics import RocCurve
from stockml.models.registry import model_label
from stockml.viz.theme import (
    BAND_FILL,
    BENCHMARK_COLOR,
    BENCHMARK_DASH,
    CATEGORICAL,
    DIVERGING_BLUE_RED,
    DOWN_COLOR,
    MODEL_COLORS,
    PRICE_COLOR,
    REFERENCE_LINE,
    SEQUENTIAL_BLUE,
    TEXT_SECONDARY,
    UP_COLOR,
    VOLUME_COLOR,
    apply_theme,
    model_color,
)

DATE_HOVER = "%{x|%d %b %Y}"
GL_THRESHOLD = 2_000
BENCHMARK_LABEL = "Buy & hold"


def _line_trace(x: pd.Index, y: pd.Series | np.ndarray, **kwargs: object) -> go.Scatter:
    """Scatter line that switches to WebGL for long series."""
    cls = go.Scattergl if len(x) > GL_THRESHOLD else go.Scatter
    trace: go.Scatter = cls(x=x, y=y, mode="lines", **kwargs)
    return trace


def downsample_ohlcv(prices: pd.DataFrame, max_points: int) -> tuple[pd.DataFrame, str]:
    """Resample daily OHLCV to weekly or monthly bars when there are too many rows to draw.

    Args:
        prices: Daily OHLCV frame.
        max_points: Largest number of bars to return.

    Returns:
        ``(frame, bar_label)`` where ``bar_label`` is ``"Daily"``, ``"Weekly"`` or ``"Monthly"``.
    """
    if len(prices) <= max_points:
        return prices, "Daily"
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    for rule, label in (("W-FRI", "Weekly"), ("ME", "Monthly")):
        resampled = prices.resample(rule).agg(agg).dropna(subset=["Close"])
        if len(resampled) <= max_points or label == "Monthly":
            return resampled, label
    raise AssertionError("unreachable")


# --- price & indicators ----------------------------------------------------------------------


def price_volume_chart(prices: pd.DataFrame, ticker: str, max_points: int = 1_500) -> go.Figure:
    """Candlestick with a volume panel sharing the date axis.

    Args:
        prices: OHLCV frame for the period to show.
        ticker: Symbol used in the title.
        max_points: Bars above this are resampled to weekly/monthly to keep the page fast.
    """
    bars, freq = downsample_ohlcv(prices, max_points)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03
    )
    fig.add_trace(
        go.Candlestick(
            x=bars.index,
            open=bars["Open"],
            high=bars["High"],
            low=bars["Low"],
            close=bars["Close"],
            name="Price",
            increasing={"line": {"color": UP_COLOR, "width": 1}, "fillcolor": UP_COLOR},
            decreasing={"line": {"color": DOWN_COLOR, "width": 1}, "fillcolor": DOWN_COLOR},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=bars.index,
            y=bars["Volume"],
            name="Volume",
            marker={"color": VOLUME_COLOR, "line": {"width": 0}},
            hovertemplate="%{y:,.0f} shares<extra>Volume</extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    apply_theme(fig, f"{ticker} price and volume ({freq.lower()} bars)", height=560)
    fig.update_layout(xaxis_rangeslider_visible=False, hovermode="x", bargap=0.1)
    fig.update_yaxes(title_text="Price (GBX)", row=1, col=1)
    fig.update_yaxes(title_text="Volume (shares)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(
        rangeselector={
            "buttons": [
                {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                {"count": 5, "label": "5Y", "step": "year", "stepmode": "backward"},
                {"step": "all", "label": "All"},
            ],
            "x": 1.0,
            "xanchor": "right",
            "y": 1.02,
            "font": {"size": 11},
        },
        row=1,
        col=1,
    )
    return fig


def indicator_chart(
    indicators: pd.DataFrame, ticker: str, config: FeatureConfig | None = None
) -> go.Figure:
    """Price with SMA/EMA/Bollinger overlays and EMA-crossover markers, plus RSI and MACD panels.

    Overlays are grouped in the legend so each family can be toggled; EMAs start hidden.

    Args:
        indicators: Output of :func:`stockml.features.pipeline.compute_indicators`.
        ticker: Symbol used in the title.
        config: Windows used to name the overlay columns.
    """
    cfg = config or FeatureConfig()
    df = indicators
    idx = df.index
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.56, 0.2, 0.24],
        vertical_spacing=0.04,
        subplot_titles=("", f"RSI ({cfg.rsi_window})", "MACD"),
    )
    fig.add_trace(
        _line_trace(
            idx,
            df["bb_upper"],
            name="Upper band",
            legendgroup="bb",
            line={"width": 0.5, "color": PRICE_COLOR},
            opacity=0.4,
            showlegend=False,
            hovertemplate="%{y:.1f}",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _line_trace(
            idx,
            df["bb_lower"],
            name=f"Bollinger ({cfg.bollinger_window}, {cfg.bollinger_num_std:g}σ)",
            legendgroup="bb",
            fill="tonexty",
            fillcolor=BAND_FILL,
            line={"width": 0.5, "color": PRICE_COLOR},
            opacity=0.4,
            hovertemplate="%{y:.1f}",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _line_trace(
            idx,
            df["Close"],
            name="Close",
            line={"width": 2, "color": TEXT_SECONDARY},
            hovertemplate="%{y:.1f} GBX",
        ),
        row=1,
        col=1,
    )
    overlay_colors = iter(CATEGORICAL[1:])
    for w in cfg.sma_windows:
        fig.add_trace(
            _line_trace(
                idx,
                df[f"sma_{w}"],
                name=f"SMA {w}",
                legendgroup="sma",
                line={"width": 1.5, "color": next(overlay_colors)},
                hovertemplate="%{y:.1f}",
            ),
            row=1,
            col=1,
        )
    for w in cfg.ema_windows:
        fig.add_trace(
            _line_trace(
                idx,
                df[f"ema_{w}"],
                name=f"EMA {w}",
                legendgroup="ema",
                visible="legendonly",
                line={"width": 1.5, "dash": "dot", "color": next(overlay_colors)},
                hovertemplate="%{y:.1f}",
            ),
            row=1,
            col=1,
        )
    short_col = f"ema_{cfg.ema_cross_short}"
    for col, symbol, color, label in (
        ("ema_cross_bullish", "triangle-up", UP_COLOR, "Bullish EMA cross"),
        ("ema_cross_bearish", "triangle-down", DOWN_COLOR, "Bearish EMA cross"),
    ):
        hits = df[df[col].astype(bool)]
        fig.add_trace(
            go.Scatter(
                x=hits.index,
                y=hits[short_col],
                mode="markers",
                name=f"{label} ({cfg.ema_cross_short}/{cfg.ema_cross_long})",
                legendgroup="cross",
                marker={
                    "symbol": symbol,
                    "size": 10,
                    "color": color,
                    "line": {"width": 1, "color": "#ffffff"},
                },
                hovertemplate="%{y:.1f}",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        _line_trace(
            idx,
            df["rsi"],
            name="RSI",
            line={"width": 1.5, "color": CATEGORICAL[6]},
            showlegend=False,
            hovertemplate="%{y:.1f}",
        ),
        row=2,
        col=1,
    )
    for level in (cfg.rsi_overbought, cfg.rsi_oversold):
        fig.add_hline(
            y=level, line={"dash": "dot", "width": 1, "color": REFERENCE_LINE}, row=2, col=1
        )
    hist = df["macd_hist"]
    fig.add_trace(
        go.Bar(
            x=idx,
            y=hist,
            name="MACD histogram",
            showlegend=False,
            marker={"color": np.where(hist >= 0, UP_COLOR, DOWN_COLOR), "line": {"width": 0}},
            hovertemplate="%{y:.2f}",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        _line_trace(
            idx,
            df["macd"],
            name="MACD line",
            showlegend=False,
            line={"width": 1.5, "color": CATEGORICAL[0]},
            hovertemplate="%{y:.2f}",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        _line_trace(
            idx,
            df["macd_signal"],
            name="Signal",
            showlegend=False,
            line={"width": 1.5, "color": CATEGORICAL[1]},
            hovertemplate="%{y:.2f}",
        ),
        row=3,
        col=1,
    )
    apply_theme(fig, f"{ticker} technical indicators", height=760)
    fig.update_layout(bargap=0, legend={"y": 1.01}, margin={"t": 150})
    fig.update_yaxes(title_text="Price (GBX)", row=1, col=1)
    fig.update_yaxes(title_text="RSI (0–100)", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD (GBX)", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_annotations(font={"size": 12, "color": TEXT_SECONDARY}, x=0, xanchor="left")
    return fig


def normalised_prices_chart(closes: pd.DataFrame) -> go.Figure:
    """Close prices of several tickers rebased to 100 at their first common date."""
    common = closes.dropna()
    rebased = common / common.iloc[0] * 100.0
    fig = go.Figure()
    for i, ticker in enumerate(rebased.columns):
        fig.add_trace(
            _line_trace(
                rebased.index,
                rebased[ticker],
                name=str(ticker),
                line={"width": 1.5, "color": CATEGORICAL[i % len(CATEGORICAL)]},
                hovertemplate="%{y:.1f}",
            ),
        )
    fig.add_hline(y=100, line={"dash": "dot", "width": 1, "color": REFERENCE_LINE})
    start = rebased.index[0].strftime("%d %b %Y") if len(rebased) else ""
    return apply_theme(
        fig, f"Relative performance (rebased to 100 on {start})", "Date", "Index level (log)"
    ).update_yaxes(type="log")


# --- exploration -----------------------------------------------------------------------------


def returns_histogram(returns: pd.DataFrame | pd.Series, title: str, bins: int = 80) -> go.Figure:
    """Overlaid histograms of *daily* returns (never cumulative), in percent.

    Args:
        returns: Daily log returns; one column per series.
        title: Chart title.
        bins: Approximate number of bins.
    """
    frame = returns.to_frame() if isinstance(returns, pd.Series) else returns
    fig = go.Figure()
    for i, col in enumerate(frame.columns):
        name = str(col)
        if name == BENCHMARK_LABEL:
            color = BENCHMARK_COLOR
        elif name in MODEL_COLORS:
            color = MODEL_COLORS[name]
        else:
            color = CATEGORICAL[i % len(CATEGORICAL)]
        fig.add_trace(
            go.Histogram(
                x=frame[col].dropna() * 100,
                name=model_label(name),
                nbinsx=bins,
                opacity=0.55,
                marker={"color": color, "line": {"width": 0}},
                hovertemplate="%{x} %<br>%{y} days<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(barmode="overlay", hovermode="closest")
    return apply_theme(fig, title, "Daily log return (%)", "Number of days")


def feature_distribution_chart(X: pd.DataFrame, y: pd.Series) -> go.Figure:
    """Per-feature histograms split by next-day outcome, with a dropdown to pick the feature.

    Args:
        X: Feature frame.
        y: Binary target aligned with ``X`` (1 = next close higher).
    """
    fig = go.Figure()
    features = list(X.columns)
    for i, feat in enumerate(features):
        for cls, label, color in ((1, "Next day up", UP_COLOR), (0, "Next day down", DOWN_COLOR)):
            values = X.loc[y == cls, feat]
            lo, hi = X[feat].quantile([0.005, 0.995])
            fig.add_trace(
                go.Histogram(
                    x=values.clip(lo, hi),
                    name=label,
                    histnorm="probability density",
                    nbinsx=60,
                    opacity=0.55,
                    visible=i == 0,
                    marker={"color": color, "line": {"width": 0}},
                    hovertemplate="%{x:.3g}<br>density %{y:.3g}<extra>%{fullData.name}</extra>",
                )
            )
    buttons = []
    for i, feat in enumerate(features):
        visible = [False] * (2 * len(features))
        visible[2 * i] = visible[2 * i + 1] = True
        buttons.append(
            {
                "label": feat,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"xaxis.title.text": f"{feat} (clipped to 0.5–99.5%)"},
                ],
            }
        )
    fig.update_layout(
        barmode="overlay",
        hovermode="closest",
        updatemenus=[
            {
                "buttons": buttons,
                "x": 1.0,
                "xanchor": "right",
                "y": 1.18,
                "yanchor": "top",
                "bgcolor": "#ffffff",
                "font": {"size": 11},
            }
        ],
    )
    return apply_theme(
        fig,
        "Feature distribution by next-day outcome",
        f"{features[0]} (clipped to 0.5–99.5%)" if features else "",
        "Density",
    )


def correlation_heatmap(X: pd.DataFrame) -> go.Figure:
    """Lower-triangle Pearson correlation heatmap of the features."""
    corr = X.corr()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    z = corr.mask(mask)
    text = z.map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    fig = go.Figure(
        go.Heatmap(
            z=z.values,
            x=list(corr.columns),
            y=list(corr.index),
            zmin=-1,
            zmax=1,
            colorscale=DIVERGING_BLUE_RED,
            text=text.values,
            texttemplate="%{text}",
            textfont={"size": 9},
            hovertemplate="%{y} × %{x}<br>ρ = %{z:.2f}<extra></extra>",
            colorbar={"title": {"text": "ρ"}, "thickness": 12},
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-45)
    return apply_theme(fig, "Feature correlation matrix", "Feature", "Feature", height=620)


def target_balance_chart(y: pd.Series) -> go.Figure:
    """Share of up days per calendar year against the 50% line (class balance over time)."""
    by_year = y.groupby(pd.DatetimeIndex(y.index).year).agg(["mean", "size"])
    fig = go.Figure(
        go.Bar(
            x=by_year.index.astype(str),
            y=by_year["mean"] * 100,
            customdata=by_year["size"],
            marker={"color": CATEGORICAL[0], "line": {"width": 0}},
            name="Up days",
            hovertemplate="%{x}: %{y:.1f}% up days (n=%{customdata})<extra></extra>",
        )
    )
    overall = float(y.mean() * 100)
    fig.add_hline(
        y=overall,
        line={"dash": "dot", "width": 1, "color": REFERENCE_LINE},
        annotation_text=f"All years {overall:.1f}%",
        annotation_position="top left",
        annotation_font={"size": 11, "color": TEXT_SECONDARY},
    )
    fig.update_layout(hovermode="closest", bargap=0.2)
    fig.update_yaxes(range=[35, 65])
    return apply_theme(
        fig, "Target balance: share of days followed by a rise", "Year", "Up days (%)"
    )


# --- models ----------------------------------------------------------------------------------


def cv_scores_chart(
    cv: Mapping[str, tuple[float, float]],
    baseline: float,
    metric_label: str = "Accuracy",
    baseline_label: str = "Naive baseline",
) -> go.Figure:
    """Mean ± std time-series CV score per model, with the naive baseline as a reference line.

    Args:
        cv: ``{model_name: (mean, std)}``.
        baseline: Baseline score for the same metric (e.g. majority-class accuracy).
        metric_label: Name of the metric for axis/title.
        baseline_label: Annotation text for the reference line.
    """
    names = list(cv)
    fig = go.Figure(
        go.Bar(
            x=[model_label(n) for n in names],
            y=[cv[n][0] for n in names],
            error_y={
                "type": "data",
                "array": [cv[n][1] for n in names],
                "color": TEXT_SECONDARY,
                "thickness": 1,
                "width": 4,
            },
            marker={"color": [model_color(n) for n in names], "line": {"width": 0}},
            hovertemplate="%{x}: %{y:.3f} ± %{error_y.array:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_hline(
        y=baseline,
        line={"dash": BENCHMARK_DASH, "width": 1.5, "color": BENCHMARK_COLOR},
        annotation_text=f"{baseline_label} {baseline:.3f}",
        annotation_position="top left",
        annotation_font={"size": 11, "color": TEXT_SECONDARY},
    )
    values = [cv[n][0] for n in names] + [baseline]
    lo, hi = min(values), max(values)
    fig.update_yaxes(range=[max(0.0, lo - 0.05), min(1.0, hi + 0.05)])
    fig.update_layout(hovermode="closest", bargap=0.35)
    return apply_theme(
        fig, f"Time-series CV {metric_label.lower()} (mean ± std)", "Model", metric_label
    )


def roc_curves_chart(curves: Mapping[str, RocCurve]) -> go.Figure:
    """ROC curves for all models on one plot with the chance diagonal."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance (AUC 0.50)",
            line={"dash": BENCHMARK_DASH, "width": 1.5, "color": BENCHMARK_COLOR},
            hoverinfo="skip",
        )
    )
    for name, curve in curves.items():
        fig.add_trace(
            go.Scatter(
                x=curve.fpr,
                y=curve.tpr,
                mode="lines",
                name=f"{model_label(name)} (AUC {curve.auc:.3f})",
                line={"width": 2, "color": model_color(name)},
                hovertemplate="FPR %{x:.2f}, TPR %{y:.2f}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(range=[0, 1], constrain="domain")
    fig.update_yaxes(range=[0, 1], scaleanchor="x", scaleratio=1)
    fig = apply_theme(
        fig, "ROC curves on the test set", "False positive rate", "True positive rate", height=560
    )
    return fig.update_layout(margin={"t": 130})


def confusion_matrix_chart(matrix: list[list[int]], name: str) -> go.Figure:
    """Confusion-matrix heatmap with counts and row percentages.

    Args:
        matrix: ``[[TN, FP], [FN, TP]]`` with rows = actual, columns = predicted.
        name: Registry name of the model.
    """
    cm = np.asarray(matrix, dtype=float)
    row_pct = cm / np.where(cm.sum(axis=1, keepdims=True) == 0, 1, cm.sum(axis=1, keepdims=True))
    labels = ["Down", "Up"]
    text = [[f"{int(cm[i, j]):,}<br>{row_pct[i, j]:.0%}" for j in range(2)] for i in range(2)]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=[f"Predicted {lab}" for lab in labels],
            y=[f"Actual {lab}" for lab in labels],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 14},
            colorscale=SEQUENTIAL_BLUE,
            showscale=False,
            xgap=2,
            ygap=2,
            hovertemplate="%{y}, %{x}: %{z:,} days<extra></extra>",
        )
    )
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="bottom")
    return apply_theme(
        fig,
        f"Confusion matrix: {model_label(name)}",
        "Predicted next-day move",
        "Actual next-day move",
        height=360,
    )


def feature_importance_chart(importance: pd.DataFrame, name: str, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of permutation importance (validation fold) with ±1 std whiskers.

    Args:
        importance: Frame indexed by feature with ``importance_mean`` and ``importance_std``.
        name: Registry name of the model.
        top_n: Number of features to show.
    """
    top = importance.sort_values("importance_mean", ascending=False).head(top_n).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=top["importance_mean"],
            y=list(top.index),
            orientation="h",
            error_x={
                "type": "data",
                "array": top["importance_std"],
                "color": TEXT_SECONDARY,
                "thickness": 1,
                "width": 3,
            },
            marker={"color": model_color(name), "line": {"width": 0}},
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line={"width": 1, "color": REFERENCE_LINE})
    fig.update_layout(hovermode="closest", bargap=0.3)
    return apply_theme(
        fig,
        f"Permutation importance: {model_label(name)}",
        "Drop in ROC AUC when shuffled (validation fold)",
        None,
        height=max(320, 26 * len(top) + 120),
    )


# --- backtest --------------------------------------------------------------------------------


def _series_style(name: str) -> dict[str, object]:
    if name == BENCHMARK_LABEL:
        return {"color": BENCHMARK_COLOR, "width": 2, "dash": BENCHMARK_DASH}
    return {"color": model_color(name), "width": 1.75}


def _time_series_chart(frame: pd.DataFrame, value_hover: str) -> go.Figure:
    fig = go.Figure()
    for col in frame.columns:
        name = str(col)
        fig.add_trace(
            _line_trace(
                frame.index,
                frame[col],
                name=model_label(name),
                line=_series_style(name),
                hovertemplate=value_hover,
            ),
        )
    return fig


def equity_curves_chart(equity: pd.DataFrame, cost_bps: float) -> go.Figure:
    """Growth of £1 for each model's strategy and buy-and-hold on one chart.

    Args:
        equity: Columns = registry names plus ``"Buy & hold"``; values = growth of 1 unit.
        cost_bps: Transaction cost used, shown in the title.
    """
    fig = _time_series_chart(equity, "£%{y:.2f}")
    fig.add_hline(y=1.0, line={"dash": "dot", "width": 1, "color": REFERENCE_LINE})
    return apply_theme(
        fig,
        f"Equity curves: growth of £1 (costs {cost_bps:g} bp per trade)",
        "Date",
        "Portfolio value (£, log scale)",
        height=480,
    ).update_yaxes(type="log")


def drawdown_chart(drawdowns: pd.DataFrame) -> go.Figure:
    """Drawdown from running peak for each strategy and buy-and-hold, in percent."""
    fig = _time_series_chart(drawdowns * 100, "%{y:.1f}%")
    return apply_theme(fig, "Drawdown from previous peak", "Date", "Drawdown (%)")


def rolling_sharpe_chart(rolling: pd.DataFrame, window: int) -> go.Figure:
    """Trailing-window annualised Sharpe ratio per strategy."""
    fig = _time_series_chart(rolling, "%{y:.2f}")
    fig.add_hline(y=0, line={"width": 1, "color": REFERENCE_LINE})
    return apply_theme(
        fig, f"Rolling {window}-day Sharpe ratio (annualised)", "Date", "Sharpe ratio"
    )


# --- multi-ticker ----------------------------------------------------------------------------


def metric_heatmap(matrix: pd.DataFrame, metric_label: str, center: float) -> go.Figure:
    """Ticker × model heatmap of one metric on a diverging scale centred on a baseline.

    Args:
        matrix: Rows = tickers, columns = registry names.
        metric_label: Metric name for the title and colour bar.
        center: Value mapped to the neutral midpoint (e.g. 0.5 accuracy or 0 Sharpe).
    """
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    spread = float(np.max(np.abs(finite - center))) if finite.size else 1.0
    spread = spread or 1.0
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=[model_label(c) for c in matrix.columns],
            y=list(matrix.index),
            zmin=center - spread,
            zmax=center + spread,
            colorscale=[[1.0 - pos, color] for pos, color in reversed(DIVERGING_BLUE_RED)],
            text=[[("" if not np.isfinite(v) else f"{v:.3f}") for v in row] for row in values],
            texttemplate="%{text}",
            textfont={"size": 11},
            xgap=2,
            ygap=2,
            colorbar={"title": {"text": metric_label}, "thickness": 12},
            hovertemplate="%{y} · %{x}<br>" + metric_label + " %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return apply_theme(
        fig,
        f"{metric_label} by ticker and model",
        "Model",
        "Ticker",
        height=max(320, 48 * len(matrix) + 180),
    )


def strategy_vs_benchmark_chart(frame: pd.DataFrame, metric_label: str) -> go.Figure:
    """Grouped bars per ticker: the selected model's metric next to buy-and-hold's.

    Args:
        frame: Index = tickers; columns ``model``, ``strategy``, ``buy_and_hold``.
        metric_label: Name of the metric plotted.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(frame.index),
            y=frame["strategy"],
            name="CV-selected model",
            customdata=[model_label(m) for m in frame["model"]],
            marker={"color": CATEGORICAL[0], "line": {"width": 0}},
            hovertemplate="%{x}: %{y:.2f} (%{customdata})<extra>CV-selected model</extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=list(frame.index),
            y=frame["buy_and_hold"],
            name=BENCHMARK_LABEL,
            marker={"color": "#c3c2b7", "line": {"width": 0}},
            hovertemplate="%{x}: %{y:.2f}<extra>" + BENCHMARK_LABEL + "</extra>",
        )
    )
    fig.add_hline(y=0, line={"width": 1, "color": REFERENCE_LINE})
    fig.update_layout(barmode="group", hovermode="closest", bargap=0.3, bargroupgap=0.08)
    return apply_theme(
        fig, f"Test-period {metric_label}: CV-selected model vs buy & hold", "Ticker", metric_label
    )
