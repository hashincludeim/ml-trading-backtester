"""Chart builders. Each function takes data and returns a themed ``plotly.graph_objects.Figure``.

No I/O and no web-framework code lives here, so the same figures render in a notebook, a
static export, or the Django dashboard. Figures are built in the light palette; dark mode is a
colour substitution (see :data:`stockml.viz.theme.DARK_COLOR_MAP`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stockml.analysis import DrawdownEpisode, binned_density, shared_edges
from stockml.config import FeatureConfig
from stockml.evaluation.metrics import RocCurve
from stockml.models.registry import model_label
from stockml.viz.theme import (
    BAND_FILL,
    BENCHMARK_COLOR,
    BENCHMARK_DASH,
    BENCHMARK_FILL,
    CATEGORICAL,
    CONTROL_ACTIVE,
    CONTROL_BG,
    DIVERGING_BLUE_RED,
    DOWN_COLOR,
    DOWN_FILL,
    MODEL_COLORS,
    PRICE_COLOR,
    REFERENCE_LINE,
    SEQUENTIAL_BLUE,
    SURFACE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    UP_COLOR,
    UP_FILL,
    apply_theme,
    date_axes,
    model_color,
)

GL_THRESHOLD = 2_000
LOG_TICKS = (0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10)  # readable log-axis steps
BENCHMARK_LABEL = "Buy & hold"
_RESAMPLE_RULES = {"Weekly": "W-FRI", "Monthly": "ME"}
_DIVERGING_REVERSED = [(1.0 - pos, color) for pos, color in reversed(DIVERGING_BLUE_RED)]
_DROPDOWN_STYLE = {
    "x": 1.0,
    "xanchor": "right",
    "y": 1.0,
    "yanchor": "bottom",
    "bgcolor": CONTROL_BG,
    "bordercolor": CONTROL_ACTIVE,
    "font": {"size": 11, "color": TEXT_PRIMARY},
    "pad": {"b": 4},
}


# --- helpers ---------------------------------------------------------------------------------


def _line_trace(x: pd.Index, y: pd.Series | np.ndarray, **kwargs: object) -> go.Scatter:
    """Scatter line that switches to WebGL for long series."""
    cls = go.Scattergl if len(x) > GL_THRESHOLD else go.Scatter
    trace: go.Scatter = cls(x=x, y=y, mode="lines", **kwargs)
    return trace


def _ref_line(fig: go.Figure, y: float, text: str | None = None) -> None:
    """Dotted horizontal reference line with an optional left-aligned label.

    Annotation kwargs are only passed when there is a label: Plotly otherwise adds a
    placeholder "new text" annotation, which on a log axis is placed at 10**y.
    """
    line = {"dash": "dot", "width": 1, "color": REFERENCE_LINE}
    if text is None:
        fig.add_hline(y=y, line=line)
        return
    fig.add_hline(
        y=y,
        line=line,
        annotation_text=text,
        annotation_position="top left",
        annotation_font={"size": 11, "color": TEXT_SECONDARY},
    )


def _add_events(
    fig: go.Figure, events: Sequence[tuple[str, str]], index: pd.Index, xref: str = "x"
) -> None:
    """Annotate dated market events that fall inside ``index`` with thin vertical markers."""
    if len(index) == 0:
        return
    first, last = pd.Timestamp(index[0]), pd.Timestamp(index[-1])
    visible = [(pd.Timestamp(d), label) for d, label in events if first <= pd.Timestamp(d) <= last]
    for i, (ts, label) in enumerate(visible):
        fig.add_shape(
            type="line",
            x0=ts,
            x1=ts,
            y0=0,
            y1=1,
            xref=xref,
            yref="y domain",
            line={"color": TEXT_MUTED, "width": 1, "dash": "dot"},
            layer="below",
        )
        fig.add_annotation(
            x=ts,
            y=1 if i % 2 == 0 else 0.91,  # stagger so neighbouring labels don't collide
            xref=xref,
            yref="y domain",
            text=label,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            xshift=3,
            font={"size": 10, "color": TEXT_SECONDARY},
            bgcolor=SURFACE,
            opacity=0.9,
        )


def _pct_axis(fig: go.Figure, axis: str = "y", **kwargs: object) -> None:
    """Format an axis whose values are fractions as percentages."""
    updater = fig.update_yaxes if axis == "y" else fig.update_xaxes
    updater(tickformat=".0%", hoverformat=".1%", **kwargs)


def _make_room_for_dropdown(fig: go.Figure) -> go.Figure:
    """Push the plot down so a top-right dropdown never overlaps the title or subtitle."""
    return fig.update_layout(margin_t=(fig.layout.margin.t or 80) + 40)


def _feature_dropdown(labels: Sequence[str], traces_per_item: int) -> list[dict[str, object]]:
    """Buttons that show the traces of one item at a time."""
    total = len(labels) * traces_per_item
    buttons: list[dict[str, object]] = []
    for i, label in enumerate(labels):
        visible = [i * traces_per_item <= j < (i + 1) * traces_per_item for j in range(total)]
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [{"visible": visible}, {"xaxis.title.text": label}],
            }
        )
    return buttons


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
    for label, rule in _RESAMPLE_RULES.items():
        resampled = prices.resample(rule).agg(agg).dropna(subset=["Close"])
        if len(resampled) <= max_points or label == "Monthly":
            return resampled, label
    raise AssertionError("unreachable")


# --- price & market history ------------------------------------------------------------------


def price_volume_chart(
    prices: pd.DataFrame,
    ticker: str,
    overlays: pd.DataFrame | None = None,
    events: Sequence[tuple[str, str]] = (),
    max_points: int = 1_500,
) -> go.Figure:
    """Candlesticks with trend overlays and event markers, plus a volume panel.

    Args:
        prices: OHLCV frame for the period to show.
        ticker: Symbol used in the title.
        overlays: Optional daily lines drawn over price (e.g. ``SMA 50``/``SMA 200``), computed
            on the full history so they have no warm-up gap.
        events: ``(date, label)`` pairs annotated when inside the range.
        max_points: Bars above this are resampled to weekly/monthly to keep the page fast.
    """
    bars, freq = downsample_ohlcv(prices, max_points)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.03
    )
    fig.add_trace(
        go.Candlestick(
            x=bars.index,
            open=bars["Open"],
            high=bars["High"],
            low=bars["Low"],
            close=bars["Close"],
            name=f"{freq} OHLC",
            increasing={"line": {"color": UP_COLOR, "width": 1}, "fillcolor": UP_COLOR},
            decreasing={"line": {"color": DOWN_COLOR, "width": 1}, "fillcolor": DOWN_COLOR},
        ),
        row=1,
        col=1,
    )
    if overlays is not None:
        lines = overlays.reindex(prices.index)
        if freq != "Daily":
            lines = lines.resample(_RESAMPLE_RULES[freq]).last().reindex(bars.index)
        for i, col in enumerate(lines.columns):
            fig.add_trace(
                _line_trace(
                    lines.index,
                    lines[col],
                    name=str(col),
                    line={"width": 1.75, "color": CATEGORICAL[(i + 3) % len(CATEGORICAL)]},
                    hovertemplate="%{y:,.1f}",
                ),
                row=1,
                col=1,
            )
    rising = (bars["Close"] >= bars["Open"]).to_numpy()
    fig.add_trace(
        go.Bar(
            x=bars.index,
            y=bars["Volume"],
            name="Volume",
            showlegend=False,
            marker={"color": np.where(rising, UP_COLOR, DOWN_COLOR), "line": {"width": 0}},
            opacity=0.45,
            hovertemplate="%{y:,.3s} shares<extra>Volume</extra>",
        ),
        row=2,
        col=1,
    )
    _add_events(fig, events, bars.index)
    apply_theme(
        fig,
        f"{ticker} price and volume",
        height=600,
        subtitle=f"{freq} candles (green = closed higher). Drag to zoom; double-click to reset.",
    )
    fig.update_layout(
        xaxis_rangeslider_visible=False, hovermode="x unified", bargap=0.05, barcornerradius=0
    )
    fig.update_yaxes(title_text="Price (GBX)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1, tickformat=".2s", showspikes=False)
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
            "y": 1.0,
            "yanchor": "bottom",
            "bgcolor": CONTROL_BG,
            "activecolor": CONTROL_ACTIVE,
            "font": {"size": 11, "color": TEXT_PRIMARY},
        },
        row=1,
        col=1,
    )
    return date_axes(fig)


def underwater_chart(
    close: pd.Series, ticker: str, episode: DrawdownEpisode | None = None
) -> go.Figure:
    """How far the price sits below its running all-time high (the "underwater" curve)."""
    dd = close / close.cummax() - 1.0
    fig = go.Figure(
        go.Scatter(
            x=dd.index,
            y=dd,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            fillcolor=DOWN_FILL,
            line={"width": 1.25, "color": DOWN_COLOR},
            hovertemplate="%{y:.1%} below peak<extra></extra>",
        )
    )
    if episode is not None:
        fig.add_annotation(
            x=episode.trough_date,
            y=-episode.depth,
            text=f"<b>−{episode.depth:.0%}</b> ({episode.trough_date:%b %Y})",
            showarrow=True,
            arrowhead=0,
            arrowcolor=TEXT_MUTED,
            ax=24,
            ay=-18,
            xanchor="left",
            font={"size": 11, "color": TEXT_PRIMARY},
            bgcolor=SURFACE,
        )
    if len(dd):
        fig.update_xaxes(range=[dd.index[0], dd.index[-1]])
    apply_theme(
        fig,
        f"{ticker} drawdown from all-time high",
        "Date",
        "Below previous peak",
        height=360,
        subtitle="0% = a new high. The deeper and wider the valley, the longer holders were down.",
    )
    _pct_axis(fig)
    fig.update_layout(showlegend=False)
    return date_axes(fig)


def annual_returns_chart(annual: pd.Series, title: str) -> go.Figure:
    """Calendar-year returns as green/red bars with value labels."""
    values = annual.to_numpy(dtype=float)
    fig = go.Figure(
        go.Bar(
            x=[str(y) for y in annual.index],
            y=values,
            name="Annual return",
            marker={"color": np.where(values >= 0, UP_COLOR, DOWN_COLOR), "line": {"width": 0}},
            text=[f"{v:+.0%}" for v in values],
            textposition="outside",
            cliponaxis=False,
            textfont={"size": 10, "color": TEXT_SECONDARY},
            hovertemplate="%{x}: %{y:+.1%}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"width": 1, "color": REFERENCE_LINE})
    apply_theme(fig, title, None, "Return", height=380)
    _pct_axis(fig)
    fig.update_layout(hovermode="closest", bargap=0.2)
    fig.update_xaxes(showspikes=False)
    return fig


def monthly_returns_heatmap(table: pd.DataFrame, title: str) -> go.Figure:
    """Year × month grid of returns (blue = gain, red = loss) with a monthly-average row."""
    grid = table.copy()
    grid.loc["Avg"] = table.mean()
    values = grid.to_numpy(dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    cap = float(np.quantile(finite, 0.95)) if finite.size else 0.1
    text = [[("" if not np.isfinite(v) else f"{v * 100:+.1f}") for v in row] for row in values]
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=list(grid.columns),
            y=[str(i) for i in grid.index],
            zmin=-cap,
            zmax=cap,
            colorscale=_DIVERGING_REVERSED,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 9},
            xgap=2,
            ygap=2,
            colorbar={"title": {"text": "Return"}, "tickformat": ".0%", "thickness": 10},
            hovertemplate="%{x} %{y}: %{z:+.1%}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False, showspikes=False, type="category")
    fig.update_xaxes(showspikes=False, side="top", showline=False)
    apply_theme(
        fig,
        title,
        None,
        None,
        height=max(380, 19 * len(grid) + 170),
        subtitle="Monthly return in %. Bottom row = average for that calendar month.",
    )
    return fig.update_layout(hovermode="closest", margin={"t": 140})


def volatility_chart(
    vol: pd.Series, ticker: str, window: int, events: Sequence[tuple[str, str]] = ()
) -> go.Figure:
    """Rolling annualised volatility with its long-run median, showing calm vs crisis regimes."""
    fig = go.Figure(
        go.Scatter(
            x=vol.index,
            y=vol,
            mode="lines",
            name="Volatility",
            fill="tozeroy",
            fillcolor=BAND_FILL,
            line={"width": 1.5, "color": PRICE_COLOR},
            hovertemplate="%{y:.0%} annualised<extra></extra>",
        )
    )
    median = float(vol.median())
    _ref_line(fig, median, f"Median {median:.0%}")
    _add_events(fig, events, vol.dropna().index)
    apply_theme(
        fig,
        f"{ticker} rolling volatility ({window}-day)",
        "Date",
        "Annualised volatility",
        height=360,
        subtitle="Spikes mark stress regimes, when daily moves are several times larger.",
    )
    _pct_axis(fig)
    fig.update_layout(showlegend=False)
    return date_axes(fig)


def normalised_prices_chart(
    closes: pd.DataFrame, events: Sequence[tuple[str, str]] = ()
) -> go.Figure:
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
                line={"width": 1.75, "color": CATEGORICAL[i % len(CATEGORICAL)]},
                hovertemplate="%{y:.1f}",
            )
        )
    _ref_line(fig, 100)
    _add_events(fig, events, rebased.index)
    start = rebased.index[0].strftime("%d %b %Y") if len(rebased) else ""
    apply_theme(
        fig,
        "Relative performance",
        "Date",
        "Index level (log scale)",
        height=460,
        legend_per_row=8,
        subtitle=f"Every stock rebased to 100 on {start}; below 100 means it lost value.",
    )
    fig.update_yaxes(type="log", tickvals=[v * 100 for v in LOG_TICKS], tickformat=".0f")
    return date_axes(fig)


# --- indicators ------------------------------------------------------------------------------


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
    rsi_title = f"RSI ({cfg.rsi_window}): momentum, 0–100"
    macd_title = "MACD: trend strength (histogram = MACD − signal)"
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.56, 0.2, 0.24],
        vertical_spacing=0.05,
        subplot_titles=("", rsi_title, macd_title),
    )
    fig.add_trace(
        _line_trace(
            idx,
            df["bb_upper"],
            name="Upper band",
            legendgroup="bb",
            showlegend=False,
            line={"width": 0.75, "color": PRICE_COLOR},
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
            line={"width": 0.75, "color": PRICE_COLOR},
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
            line={"width": 2.25, "color": TEXT_PRIMARY},
            hovertemplate="<b>%{y:.1f} GBX</b>",
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
                    "size": 11,
                    "color": color,
                    "line": {"width": 1.5, "color": SURFACE},
                },
                hovertemplate=label,
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
            hovertemplate="RSI %{y:.1f}",
        ),
        row=2,
        col=1,
    )
    # Shapes must be added after the subplot has a trace, or Plotly skips them.
    fig.add_hrect(y0=cfg.rsi_overbought, y1=100, fillcolor=DOWN_FILL, line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=cfg.rsi_oversold, fillcolor=UP_FILL, line_width=0, row=2, col=1)
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
            opacity=0.7,
            hovertemplate="hist %{y:+.2f}",
        ),
        row=3,
        col=1,
    )
    for column, label, color in (
        ("macd", "MACD", CATEGORICAL[0]),
        ("macd_signal", "Signal", CATEGORICAL[1]),
    ):
        fig.add_trace(
            _line_trace(
                idx,
                df[column],
                name=label,
                showlegend=False,
                line={"width": 1.5, "color": color},
                hovertemplate=label + " %{y:.2f}",
            ),
            row=3,
            col=1,
        )
    apply_theme(
        fig,
        f"{ticker} technical indicators",
        height=800,
        subtitle="Click legend items to toggle overlays. RSI shading: overbought (red) and "
        "oversold (green).",
    )
    fig.update_layout(bargap=0, barcornerradius=0, legend={"y": 1.01}, margin={"t": 170})
    fig.update_yaxes(title_text="Price (GBX)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], tickvals=[0, 30, 50, 70, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD (GBX)", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    for text in (rsi_title, macd_title):
        fig.update_annotations(
            selector={"text": text},
            font={"size": 12, "color": TEXT_SECONDARY},
            x=0,
            xanchor="left",
        )
    return date_axes(fig)


def indicator_signal_chart(table: pd.DataFrame, base_rate: float) -> go.Figure:
    """Dot plot of the next-day up-rate after each indicator state, with 95% intervals.

    Args:
        table: Output of :func:`stockml.analysis.indicator_signal_table`.
        base_rate: Overall share of up days.
    """
    fig = go.Figure()
    families = list(dict.fromkeys(table["indicator"]))
    for i, family in enumerate(families):
        rows = table[table["indicator"] == family]
        fig.add_trace(
            go.Scatter(
                x=rows["up_rate"],
                y=[f"{family} · {s}" for s in rows["state"]],
                mode="markers",
                name=family,
                marker={
                    "size": 12,
                    "color": CATEGORICAL[i % len(CATEGORICAL)],
                    "line": {"width": 1.5, "color": SURFACE},
                },
                error_x={
                    "type": "data",
                    "array": 1.96 * rows["se"],
                    "color": TEXT_MUTED,
                    "thickness": 1.25,
                    "width": 5,
                },
                customdata=np.column_stack([rows["n"], rows["mean_return"] * 10_000]),
                hovertemplate="%{y}<br>up next day %{x:.1%} · n=%{customdata[0]:,}"
                "<br>avg next-day return %{customdata[1]:+.1f} bp<extra></extra>",
            )
        )
    fig.add_vline(x=base_rate, line={"dash": "dot", "width": 1, "color": REFERENCE_LINE})
    fig.add_annotation(
        x=base_rate,
        y=1,
        yref="paper",
        text=f"Average {base_rate:.1%}",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        xshift=4,
        font={"size": 11, "color": TEXT_SECONDARY},
    )
    apply_theme(
        fig,
        "What happened the day after each signal?",
        "Share of next days that closed higher",
        None,
        height=max(400, 34 * len(table) + 170),
        subtitle="Dots on the dotted line = the signal told you nothing. Whiskers = 95% range.",
    )
    _pct_axis(fig, axis="x")
    fig.update_xaxes(showgrid=True, showspikes=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, showspikes=False)
    return fig.update_layout(hovermode="closest")


# --- exploration -----------------------------------------------------------------------------


def returns_histogram(
    returns: pd.DataFrame | pd.Series, title: str, bins: int = 80, fit_normal: bool = False
) -> go.Figure:
    """Overlaid, pre-binned histograms of *daily* returns (never cumulative), in percent.

    Args:
        returns: Daily log returns; one column per series.
        title: Chart title.
        bins: Number of bins over the central 99% of values.
        fit_normal: Overlay a normal curve with the first column's mean and std, which makes fat
            tails visible.
    """
    frame = returns.to_frame() if isinstance(returns, pd.Series) else returns
    series = [frame[c].dropna() * 100 for c in frame.columns]
    edges = shared_edges(series, bins)
    width = float(edges[1] - edges[0])
    fig = go.Figure()
    for i, (col, values) in enumerate(zip(frame.columns, series, strict=True)):
        name = str(col)
        if name == BENCHMARK_LABEL:
            color = BENCHMARK_COLOR
        elif name in MODEL_COLORS:
            color = MODEL_COLORS[name]
        else:
            color = CATEGORICAL[i % len(CATEGORICAL)]
        counts, _ = np.histogram(values.clip(edges[0], edges[-1]), bins=edges)
        fig.add_trace(
            go.Bar(
                x=(edges[:-1] + edges[1:]) / 2,
                y=counts,
                width=width,
                name=model_label(name),
                marker={"color": color, "line": {"width": 0}},
                opacity=0.55,
                hovertemplate="%{x:.2f}%: %{y} days<extra>%{fullData.name}</extra>",
            )
        )
    if fit_normal and series:
        base = series[0]
        mu, sd = float(base.mean()), float(base.std())
        xs = np.linspace(edges[0], edges[-1], 400)
        pdf = np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=pdf * len(base) * width,
                mode="lines",
                name="Normal distribution (same mean & std)",
                line={"color": TEXT_PRIMARY, "width": 1.5, "dash": "dash"},
                hoverinfo="skip",
            )
        )
    fig.update_layout(barmode="overlay", hovermode="closest", bargap=0, barcornerradius=0)
    fig.update_xaxes(ticksuffix="%", showspikes=False)
    return apply_theme(
        fig,
        title,
        "Daily log return",
        "Number of days",
        subtitle="Outer 0.5% on each side folded into the end bins.",
    )


def feature_distribution_chart(X: pd.DataFrame, y: pd.Series, bins: int = 50) -> go.Figure:
    """Per-feature density for days followed by a rise vs a fall, with a feature dropdown.

    Densities are pre-binned (small payload) and drawn as filled step lines so the two outcomes
    stay distinguishable where they overlap.
    """
    fig = go.Figure()
    features = list(X.columns)
    for i, feat in enumerate(features):
        edges = shared_edges([X[feat]], bins)
        for cls, label, color, fill in (
            (1, "Next day up", UP_COLOR, UP_FILL),
            (0, "Next day down", DOWN_COLOR, DOWN_FILL),
        ):
            values = X.loc[y == cls, feat].clip(edges[0], edges[-1])
            centres, density = binned_density(values, edges)
            fig.add_trace(
                go.Scatter(
                    x=centres,
                    y=density,
                    name=f"{label} (n={len(values):,})",
                    mode="lines",
                    line={"shape": "hvh", "width": 1.75, "color": color},
                    fill="tozeroy",
                    fillcolor=fill,
                    visible=i == 0,
                    hovertemplate="density %{y:.3g}<extra>" + label + "</extra>",
                )
            )
    fig.update_layout(
        hovermode="x unified",
        updatemenus=[{"buttons": _feature_dropdown(features, 2), **_DROPDOWN_STYLE}],
    )
    apply_theme(
        fig,
        "Feature distribution by next-day outcome",
        features[0] if features else "",
        "Density",
        height=470,
        subtitle="Near-identical shapes mean the feature alone can't separate up days from down.",
    )
    return _make_room_for_dropdown(fig)


def feature_signal_chart(
    stats: Mapping[str, pd.DataFrame], base_rate: float, n_bins: int
) -> go.Figure:
    """Next-day up-rate by quantile bucket of each feature (dropdown), with 95% error bars.

    Args:
        stats: ``{feature: feature_bucket_stats(...)}``.
        base_rate: Overall share of up days (reference line).
        n_bins: Requested bucket count, for the axis label.
    """
    fig = go.Figure()
    features = list(stats)
    for i, feat in enumerate(features):
        s = stats[feat]
        labels = [f"Q{k + 1}" for k in range(len(s))]
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=s["up_rate"],
                mode="lines+markers",
                name=feat,
                visible=i == 0,
                line={"width": 1.5, "color": REFERENCE_LINE},
                marker={
                    "size": 10,
                    "color": np.where(s["up_rate"] >= base_rate, UP_COLOR, DOWN_COLOR),
                    "line": {"width": 1.5, "color": SURFACE},
                },
                error_y={
                    "type": "data",
                    "array": 1.96 * s["se"],
                    "color": TEXT_MUTED,
                    "thickness": 1,
                    "width": 4,
                },
                customdata=np.column_stack([s["lower"], s["upper"], s["n"]]),
                hovertemplate="%{x}: %{customdata[0]:.3g} to %{customdata[1]:.3g}<br>"
                "next day up %{y:.1%} (n=%{customdata[2]:,})<extra></extra>",
            )
        )
    _ref_line(fig, base_rate, f"Average {base_rate:.1%}")
    buttons = _feature_dropdown(features, 1)
    for button, feat in zip(buttons, features, strict=True):
        button["args"] = [button["args"][0], {"xaxis.title.text": f"{feat} bucket"}]  # type: ignore[index]
    fig.update_layout(
        hovermode="closest",
        showlegend=False,
        updatemenus=[{"buttons": buttons, **_DROPDOWN_STYLE}],
    )
    apply_theme(
        fig,
        "Does the feature predict tomorrow?",
        f"{features[0]} bucket" if features else "",
        "Share of next days up",
        height=470,
        subtitle=f"Days split into {n_bins} equal groups by feature value (Q1 = lowest).",
    )
    _pct_axis(fig, range=[0.38, 0.62])
    fig.update_xaxes(showspikes=False)
    return _make_room_for_dropdown(fig)


def feature_correlation_chart(corr: pd.Series, band: float) -> go.Figure:
    """Rank correlation of each feature with the next-day return, against a noise band."""
    ordered = corr.iloc[::-1]
    values = ordered.to_numpy(dtype=float)
    fig = go.Figure(
        go.Bar(
            x=values,
            y=list(ordered.index),
            orientation="h",
            marker={
                "color": np.where(np.abs(values) > band, CATEGORICAL[0], BENCHMARK_FILL),
                "line": {"width": 0},
            },
            hovertemplate="%{y}: ρ = %{x:+.3f}<extra></extra>",
        )
    )
    fig.add_vrect(x0=-band, x1=band, fillcolor=BAND_FILL, line_width=0, layer="below")
    fig.add_annotation(
        x=0,
        y=1,
        yref="paper",
        text=f"noise band ±{band:.3f}",
        showarrow=False,
        yanchor="bottom",
        font={"size": 10, "color": TEXT_SECONDARY},
    )
    fig.add_vline(x=0, line={"width": 1, "color": REFERENCE_LINE})
    apply_theme(
        fig,
        "Correlation with next-day return",
        "Spearman rank correlation ρ",
        None,
        height=max(380, 22 * len(values) + 170),
        subtitle="Grey bars sit inside the shaded 95% noise band: no reliable signal.",
    )
    fig.update_layout(hovermode="closest", bargap=0.3, margin={"t": 130})
    fig.update_xaxes(showgrid=True, showspikes=False)
    fig.update_yaxes(showgrid=False, showspikes=False)
    return fig


def autocorrelation_chart(acf: pd.Series, band: float, ticker: str) -> go.Figure:
    """Autocorrelation of daily returns by lag with a 95% noise band."""
    values = acf.to_numpy(dtype=float)
    outside = np.abs(values) > band
    fig = go.Figure(
        go.Bar(
            x=list(acf.index),
            y=values,
            name="Autocorrelation",
            marker={
                "color": np.where(outside, CATEGORICAL[1], BENCHMARK_FILL),
                "line": {"width": 0},
            },
            hovertemplate="lag %{x} days: %{y:+.3f}<extra></extra>",
        )
    )
    fig.add_hrect(y0=-band, y1=band, fillcolor=BAND_FILL, line_width=0, layer="below")
    fig.add_hline(y=0, line={"width": 1, "color": REFERENCE_LINE})
    apply_theme(
        fig,
        f"{ticker}: do past returns predict future returns?",
        "Lag (trading days)",
        "Autocorrelation",
        height=380,
        subtitle="Shaded = 95% band if days were independent. Orange bars fall outside it.",
    )
    fig.update_layout(hovermode="closest", bargap=0.35)
    fig.update_xaxes(dtick=1, showspikes=False)
    return fig


def correlation_heatmap(
    frame: pd.DataFrame, title: str = "Feature correlation matrix", subtitle: str | None = None
) -> go.Figure:
    """Lower-triangle Pearson correlation heatmap of the columns of ``frame``."""
    corr = frame.corr()
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
            colorscale=_DIVERGING_REVERSED,
            text=text.values,
            texttemplate="%{text}",
            textfont={"size": 9 if len(corr) > 8 else 12},
            hovertemplate="%{y} × %{x}<br>ρ = %{z:.2f}<extra></extra>",
            colorbar={"title": {"text": "ρ"}, "thickness": 10},
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False, showspikes=False, showline=False)
    fig.update_xaxes(showgrid=False, showspikes=False, showline=False, tickangle=-45)
    return apply_theme(
        fig,
        title,
        None,
        None,
        height=max(380, 30 * len(corr) + 190),
        subtitle=subtitle or "Blue = move together, red = move opposite. Lower triangle only.",
    )


def target_balance_chart(y: pd.Series) -> go.Figure:
    """Share of up days per calendar year against the overall average (class balance)."""
    by_year = y.groupby(pd.DatetimeIndex(y.index).year).agg(["mean", "size"])
    overall = float(y.mean())
    fig = go.Figure(
        go.Bar(
            x=by_year.index.astype(str),
            y=by_year["mean"],
            customdata=by_year["size"],
            marker={"color": CATEGORICAL[0], "line": {"width": 0}},
            name="Up days",
            hovertemplate="%{x}: %{y:.1%} up days (n=%{customdata})<extra></extra>",
        )
    )
    _ref_line(fig, overall, f"All years {overall:.1%}")
    fig.update_layout(hovermode="closest", bargap=0.2)
    apply_theme(
        fig,
        "Target balance: share of days followed by a rise",
        None,
        "Up days",
        height=380,
        subtitle="Close to 50% every year, so accuracy must be judged against ~50%, not 0%.",
    )
    _pct_axis(fig, range=[0.38, 0.62])
    fig.update_xaxes(showspikes=False)
    return fig


# --- models ----------------------------------------------------------------------------------


def cv_scores_chart(
    cv: Mapping[str, tuple[float, float]],
    baseline: float,
    metric_label: str = "Accuracy",
    baseline_label: str = "Naive baseline",
) -> go.Figure:
    """Mean ± std time-series CV score per model, with a baseline reference line.

    Args:
        cv: ``{model_name: (mean, std)}``.
        baseline: Baseline score for the same metric (e.g. 0.5 for a coin flip).
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
            hovertemplate="%{x}: %{y:.1%} ± %{error_y.array:.1%}<extra></extra>",
            showlegend=False,
        )
    )
    _ref_line(fig, baseline, f"{baseline_label} {baseline:.1%}")
    values = [cv[n][0] for n in names] + [baseline]
    lo, hi = min(values), max(values)
    fig.update_layout(hovermode="closest", bargap=0.35)
    apply_theme(
        fig,
        f"Time-series CV {metric_label.lower()} (mean ± std)",
        None,
        metric_label,
        subtitle="Scored on expanding-window folds of the training period only.",
    )
    _pct_axis(fig, range=[max(0.0, lo - 0.05), min(1.0, hi + 0.05)])
    fig.update_xaxes(showspikes=False)
    return fig


def cv_folds_chart(
    scores: Mapping[str, Sequence[float]], metric_label: str = "Accuracy"
) -> go.Figure:
    """Score on each chronological CV fold per model: shows how unstable any edge is."""
    fig = go.Figure()
    for name, values in scores.items():
        fig.add_trace(
            go.Scatter(
                x=[f"Fold {i + 1}" for i in range(len(values))],
                y=list(values),
                mode="lines+markers",
                name=model_label(name),
                line={"width": 1.75, "color": model_color(name)},
                marker={"size": 8, "line": {"width": 1.5, "color": SURFACE}},
                hovertemplate="%{y:.1%}",
            )
        )
    _ref_line(fig, 0.5, "Coin flip")
    apply_theme(
        fig,
        f"CV {metric_label.lower()} fold by fold",
        "Chronological fold (1 = earliest)",
        metric_label,
        height=440,
        subtitle="Lines crossing 50% from fold to fold mean the edge is not stable over time.",
    )
    _pct_axis(fig)
    fig.update_xaxes(showspikes=False)
    return fig


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
    fig.update_xaxes(range=[0, 1], constrain="domain", showgrid=True, showspikes=False)
    fig.update_yaxes(range=[0, 1], scaleanchor="x", scaleratio=1, showspikes=False)
    apply_theme(
        fig,
        "ROC curves on the test set",
        "False positive rate",
        "True positive rate",
        height=600,
        subtitle="Curves hugging the dashed diagonal are no better than guessing.",
    )
    return fig


def confusion_matrix_chart(matrix: list[list[int]], name: str) -> go.Figure:
    """Confusion-matrix heatmap with counts and row percentages.

    Args:
        matrix: ``[[TN, FP], [FN, TP]]`` with rows = actual, columns = predicted.
        name: Registry name of the model.
    """
    cm = np.asarray(matrix, dtype=float)
    totals = cm.sum(axis=1, keepdims=True)
    row_pct = cm / np.where(totals == 0, 1, totals)
    labels = ["Down", "Up"]
    text = [
        [
            f"<b>{int(cm[i, j]):,}</b><br>{row_pct[i, j]:.0%} of actual {labels[i].lower()}"
            for j in range(2)
        ]
        for i in range(2)
    ]
    fig = go.Figure(
        go.Heatmap(
            z=row_pct,
            x=[f"Predicted {lab}" for lab in labels],
            y=[f"Actual {lab}" for lab in labels],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 13},
            colorscale=SEQUENTIAL_BLUE,
            zmin=0,
            zmax=1,
            showscale=False,
            xgap=3,
            ygap=3,
            customdata=cm,
            hovertemplate="%{y}, %{x}: %{customdata:,} days (%{z:.0%} of row)<extra></extra>",
        )
    )
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False, showspikes=False, showline=False)
    fig.update_xaxes(showgrid=False, side="bottom", showspikes=False, showline=False)
    return apply_theme(
        fig,
        f"Confusion matrix: {model_label(name)}",
        "Predicted next-day move",
        "Actual next-day move",
        height=400,
        subtitle="Shading = share of each actual outcome. A good model is dark on the diagonal.",
    )


def feature_importance_chart(importance: pd.DataFrame, name: str, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of permutation importance (validation fold) with ±1 std whiskers.

    Args:
        importance: Frame indexed by feature with ``importance_mean`` and ``importance_std``.
        name: Registry name of the model.
        top_n: Number of features to show.
    """
    top = importance.sort_values("importance_mean", ascending=False).head(top_n).iloc[::-1]
    values = top["importance_mean"].to_numpy(dtype=float)
    fig = go.Figure(
        go.Bar(
            x=values,
            y=list(top.index),
            orientation="h",
            error_x={
                "type": "data",
                "array": top["importance_std"],
                "color": TEXT_SECONDARY,
                "thickness": 1,
                "width": 3,
            },
            marker={
                "color": np.where(values > 0, model_color(name), BENCHMARK_FILL),
                "line": {"width": 0},
            },
            hovertemplate="%{y}: %{x:+.4f} AUC<extra></extra>",
        )
    )
    fig.add_vline(x=0, line={"width": 1, "color": REFERENCE_LINE})
    fig.update_layout(hovermode="closest", bargap=0.3)
    fig.update_xaxes(showgrid=True, showspikes=False)
    fig.update_yaxes(showgrid=False, showspikes=False)
    return apply_theme(
        fig,
        f"Permutation importance: {model_label(name)}",
        "Drop in ROC AUC when shuffled (validation fold)",
        None,
        height=max(360, 26 * len(top) + 150),
        subtitle="Grey bars (≤ 0): shuffling that feature didn't hurt, so the model ignores it.",
    )


def score_distribution_chart(
    scores: pd.Series, y_true: pd.Series, name: str, bins: int = 40
) -> go.Figure:
    """Distribution of a model's test-set scores, split by what actually happened next day."""
    edges = shared_edges([scores], bins)
    fig = go.Figure()
    for cls, label, color, fill in (
        (1, "Actually went up", UP_COLOR, UP_FILL),
        (0, "Actually went down", DOWN_COLOR, DOWN_FILL),
    ):
        values = scores[y_true == cls].clip(edges[0], edges[-1])
        centres, density = binned_density(values, edges)
        fig.add_trace(
            go.Scatter(
                x=centres,
                y=density,
                mode="lines",
                name=f"{label} (n={len(values):,})",
                line={"shape": "hvh", "width": 1.75, "color": color},
                fill="tozeroy",
                fillcolor=fill,
                hovertemplate="density %{y:.3g}",
            )
        )
    apply_theme(
        fig,
        f"How well did {model_label(name)} separate the days?",
        "Model score (higher = more bullish)",
        "Density",
        height=420,
        subtitle="If the model could tell days apart, the green and red shapes would separate.",
    )
    fig.update_xaxes(showspikes=False)
    return fig.update_layout(hovermode="x unified")


def rolling_accuracy_chart(hit_rates: pd.DataFrame, window: int) -> go.Figure:
    """Trailing-window accuracy of each model through the test period."""
    fig = go.Figure()
    for col in hit_rates.columns:
        fig.add_trace(
            _line_trace(
                hit_rates.index,
                hit_rates[col],
                name=model_label(str(col)),
                line={"width": 1.5, "color": model_color(str(col))},
                hovertemplate="%{y:.1%}",
            )
        )
    _ref_line(fig, 0.5, "Coin flip")
    apply_theme(
        fig,
        f"Rolling {window}-day accuracy on the test set",
        "Date",
        "Accuracy",
        height=440,
        subtitle="A lasting edge would keep a line above 50% throughout the test period.",
    )
    _pct_axis(fig)
    return date_axes(fig)


# --- backtest --------------------------------------------------------------------------------


def _series_style(name: str) -> dict[str, object]:
    if name == BENCHMARK_LABEL:
        return {"color": BENCHMARK_COLOR, "width": 2.25, "dash": BENCHMARK_DASH}
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
            )
        )
    return fig


def equity_curves_chart(equity: pd.DataFrame, cost_bps: float) -> go.Figure:
    """Growth of £1 for each model's strategy and buy-and-hold on one chart.

    Args:
        equity: Columns = registry names plus ``"Buy & hold"``; values = growth of 1 unit.
        cost_bps: Transaction cost used, shown in the title.
    """
    fig = _time_series_chart(equity, "£%{y:.2f}")
    _ref_line(fig, 1.0)
    fig.add_annotation(  # log axis: annotation y is log10(value), so £1 -> 0
        x=1,
        xref="paper",
        y=0,
        text="Break-even £1",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font={"size": 11, "color": TEXT_SECONDARY},
    )
    apply_theme(
        fig,
        f"Growth of £1 (after {cost_bps:g} bp costs per trade)",
        "Date",
        "Portfolio value (log scale)",
        height=520,
        legend_per_row=8,
        subtitle="Dashed grey = simply holding the stock. Click a legend entry to hide it.",
    )
    fig.update_yaxes(type="log", tickvals=LOG_TICKS, tickformat=".2~f", tickprefix="£")
    return date_axes(fig)


def drawdown_chart(drawdowns: pd.DataFrame) -> go.Figure:
    """Drawdown from running peak for each strategy and buy-and-hold."""
    fig = _time_series_chart(drawdowns, "%{y:.1%}")
    apply_theme(
        fig,
        "Drawdown from previous peak",
        "Date",
        "Below peak",
        height=440,
        subtitle="How much each strategy had lost from its best point at every date.",
    )
    _pct_axis(fig)
    return date_axes(fig)


def rolling_sharpe_chart(rolling: pd.DataFrame, window: int) -> go.Figure:
    """Trailing-window annualised Sharpe ratio per strategy."""
    fig = _time_series_chart(rolling, "%{y:.2f}")
    fig.add_hline(y=0, line={"width": 1, "color": REFERENCE_LINE})
    apply_theme(
        fig,
        f"Rolling {window}-day Sharpe ratio",
        "Date",
        "Sharpe (annualised)",
        height=440,
        subtitle="Above 0 = earning more than it risked over the trailing window.",
    )
    return date_axes(fig)


def cost_sensitivity_chart(sharpes: pd.DataFrame, current_cost: float) -> go.Figure:
    """Sharpe ratio of every strategy as transaction costs rise, with buy-and-hold for reference.

    Args:
        sharpes: Output of :func:`stockml.evaluation.backtest.cost_sensitivity`; the
            ``buy_and_hold`` column is drawn as the benchmark.
        current_cost: Cost selected on the page, marked with a vertical line.
    """
    frame = sharpes.rename(columns={"buy_and_hold": BENCHMARK_LABEL})
    fig = go.Figure()
    for col in frame.columns:
        name = str(col)
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[col],
                mode="lines+markers",
                name=model_label(name),
                line=_series_style(name),
                marker={"size": 6},
                hovertemplate="%{y:.2f}",
            )
        )
    fig.add_hline(y=0, line={"width": 1, "color": REFERENCE_LINE})
    fig.add_vline(x=current_cost, line={"dash": "dot", "width": 1, "color": TEXT_MUTED})
    fig.add_annotation(
        x=current_cost,
        y=0,
        yref="paper",
        text="selected cost",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        xshift=3,
        font={"size": 10, "color": TEXT_SECONDARY},
    )
    apply_theme(
        fig,
        "How trading costs erode each strategy",
        "Cost per trade (basis points)",
        "Sharpe ratio",
        height=460,
        subtitle="Daily strategies flip position often, so every basis point of cost compounds.",
    )
    fig.update_xaxes(showgrid=True, ticksuffix=" bp")
    return fig


def risk_return_scatter(points: pd.DataFrame, title: str, subtitle: str) -> go.Figure:
    """Annualised volatility vs return, one labelled marker per strategy or ticker.

    Args:
        points: Index = display labels; columns ``annual_volatility``, ``annual_return``,
            ``sharpe``, ``color`` and optional boolean ``benchmark``.
        title: Chart title.
        subtitle: One-line reading hint.
    """
    fig = go.Figure()
    ordered = points.sort_values("annual_return", ascending=False)
    for rank, (label, row) in enumerate(ordered.iterrows()):
        is_bench = bool(row.get("benchmark", False))
        fig.add_trace(
            go.Scatter(
                x=[row["annual_volatility"]],
                y=[row["annual_return"]],
                mode="markers+text",
                name=str(label),
                text=[str(label)],
                textposition="middle right" if rank % 2 == 0 else "middle left",
                textfont={"size": 11, "color": TEXT_SECONDARY},
                marker={
                    "size": 13,
                    "color": BENCHMARK_FILL if is_bench else row["color"],
                    "symbol": "diamond" if is_bench else "circle",
                    "line": {"width": 1.5, "color": BENCHMARK_COLOR if is_bench else SURFACE},
                },
                customdata=[[row["sharpe"]]],
                hovertemplate="<b>%{text}</b><br>return %{y:+.1%} a year<br>volatility %{x:.1%}"
                "<br>Sharpe %{customdata[0]:.2f}<extra></extra>",
                showlegend=False,
            )
        )
    fig.add_hline(y=0, line={"width": 1, "color": REFERENCE_LINE})
    apply_theme(
        fig,
        title,
        "Annualised volatility (risk)",
        "Annualised return",
        height=460,
        subtitle=subtitle,
    )
    _pct_axis(fig)
    _pct_axis(fig, axis="x")
    vols = points["annual_volatility"].astype(float)
    pad = max(float(vols.max() - vols.min()) * 0.35, 0.04)
    fig.update_xaxes(
        showgrid=True, showspikes=False, range=[float(vols.min()) - pad, float(vols.max()) + pad]
    )
    fig.update_yaxes(showspikes=False)
    return fig.update_layout(hovermode="closest")


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
            colorscale=_DIVERGING_REVERSED,
            text=[[("" if not np.isfinite(v) else f"{v:.3f}") for v in row] for row in values],
            texttemplate="%{text}",
            textfont={"size": 11},
            xgap=2,
            ygap=2,
            colorbar={"title": {"text": metric_label}, "thickness": 10},
            hovertemplate="%{y} · %{x}<br>" + metric_label + " %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(showgrid=False, showspikes=False, showline=False)
    fig.update_yaxes(showgrid=False, autorange="reversed", showspikes=False, showline=False)
    neutral = "zero" if center == 0 else f"{center:g} (chance)"
    return apply_theme(
        fig,
        f"{metric_label} by ticker and model",
        None,
        None,
        height=max(340, 48 * len(matrix) + 200),
        subtitle=f"Colour centred on {neutral}: blue is better, red is worse.",
    )


def strategy_vs_benchmark_chart(frame: pd.DataFrame, metric_label: str) -> go.Figure:
    """Grouped bars per ticker: the selected model's metric next to buy-and-hold's.

    Args:
        frame: Index = tickers; columns ``model``, ``strategy``, ``buy_and_hold``.
        metric_label: Name of the metric plotted.
    """
    fig = go.Figure()
    for column, name, color in (
        ("strategy", "CV-selected model", CATEGORICAL[0]),
        ("buy_and_hold", BENCHMARK_LABEL, BENCHMARK_FILL),
    ):
        fig.add_trace(
            go.Bar(
                x=list(frame.index),
                y=frame[column],
                name=name,
                customdata=[model_label(m) for m in frame["model"]],
                marker={"color": color, "line": {"width": 0}},
                text=[f"{v:.2f}" for v in frame[column]],
                textposition="outside",
                textfont={"size": 10, "color": TEXT_SECONDARY},
                cliponaxis=False,
                hovertemplate="%{x}: %{y:.2f}"
                + (" (%{customdata})" if column == "strategy" else "")
                + f"<extra>{name}</extra>",
            )
        )
    fig.add_hline(y=0, line={"width": 1, "color": REFERENCE_LINE})
    fig.update_layout(barmode="group", hovermode="closest", bargap=0.3, bargroupgap=0.08)
    fig.update_xaxes(showspikes=False)
    return apply_theme(
        fig,
        f"Test-period {metric_label}: CV-selected model vs buy & hold",
        None,
        metric_label,
        subtitle="The model is chosen on training data only, so this is a fair out-of-sample test.",
    )
