"""Shared visual language for every chart: palette, fonts, hover, margins.

Categorical colours follow a fixed, colour-blind-validated order and are bound to *model
names*, so a model keeps its colour on every page regardless of which other models are shown.
"""

from __future__ import annotations

import math

import plotly.graph_objects as go
import plotly.io as pio

from stockml.models.registry import MODEL_REGISTRY

# Matches the dashboard's UI face (dashboard.css), so chart text and page text are one family.
FONT_FAMILY = "'Instrument Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

# Neutral chrome. SURFACE is also the page colour, so charts sit on the page without a box.
SURFACE = "#ffffff"
SURFACE_RAISED = "#fafafa"  # hover labels
TEXT_PRIMARY = "#111113"
TEXT_SECONDARY = "#4a4a50"
TEXT_MUTED = "#6e6e76"  # 5:1 on SURFACE, so tick labels stay readable
GRID = "#ebebee"
AXIS_LINE = "#c4c4ca"
CONTROL_BG = "#f3f3f4"
CONTROL_ACTIVE = "#dcdce0"

# Categorical slots, in validated order (identity only, never rank).
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
# The same eight hues stepped for a dark surface (validated as a set, same order).
CATEGORICAL_DARK: tuple[str, ...] = (
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
)

MODEL_COLORS: dict[str, str] = {
    name: CATEGORICAL[i % len(CATEGORICAL)] for i, name in enumerate(MODEL_REGISTRY)
}
BENCHMARK_COLOR = TEXT_SECONDARY  # buy-and-hold / baselines: neutral ink, dashed
BENCHMARK_FILL = "#c9c9ce"  # benchmark as a filled mark (bars)
BENCHMARK_DASH = "dash"

UP_COLOR = "#1baf7a"
DOWN_COLOR = "#e34948"
UP_FILL = "rgba(27, 175, 122, 0.12)"
DOWN_FILL = "rgba(227, 73, 72, 0.14)"
# Volatility target classes: orange for a big move, blue for a quiet day (not good/bad colours).
BIG_MOVE_COLOR = "#eb6834"
QUIET_COLOR = "#2a78d6"
BIG_MOVE_FILL = "rgba(235, 104, 52, 0.12)"
QUIET_FILL = "rgba(42, 120, 214, 0.12)"
PRICE_COLOR = "#2a78d6"
BAND_FILL = "rgba(42, 120, 214, 0.08)"
REFERENCE_LINE = TEXT_MUTED
# Evaluation timeline: rows a model learns from, the folds CV scores it on, the unseen test.
SPLIT_TRAIN = BENCHMARK_FILL
SPLIT_VALIDATION = CATEGORICAL[0]
SPLIT_TEST = CATEGORICAL[1]

SEQUENTIAL_BLUE: list[tuple[float, str]] = [
    (0.0, "#f3f8fe"),
    (0.25, "#b7d3f6"),
    (0.5, "#6da7ec"),
    (0.75, "#256abf"),
    (1.0, "#0d366b"),
]
DIVERGING_MIDPOINT = "#f0efec"  # neutral grey: "no change" reads as nothing
DIVERGING_BLUE_RED: list[tuple[float, str]] = [
    (0.0, "#184f95"),
    (0.25, "#6da7ec"),
    (0.5, DIVERGING_MIDPOINT),
    (0.75, "#ee8a86"),
    (1.0, "#a8262a"),
]

# Light -> dark substitutions. Figures are built in light mode; the browser swaps every colour
# found here when the dark theme is active, so both modes come from this one module. The dark
# SURFACE must match the dark page colour in dashboard.css.
DARK_COLOR_MAP: dict[str, str] = {
    SURFACE: "#111113",
    SURFACE_RAISED: "#1c1c1f",
    TEXT_PRIMARY: "#f1f1ef",
    TEXT_SECONDARY: "#b9b9be",
    TEXT_MUTED: "#8b8b92",
    GRID: "#222226",
    AXIS_LINE: "#46464c",
    CONTROL_BG: "#27272b",
    CONTROL_ACTIVE: "#3a3a3f",
    BENCHMARK_FILL: "#4d4d53",
    DIVERGING_MIDPOINT: "#383835",
    UP_FILL: "rgba(25, 158, 112, 0.22)",
    DOWN_FILL: "rgba(230, 103, 103, 0.24)",
    BAND_FILL: "rgba(57, 135, 229, 0.16)",
    BIG_MOVE_FILL: "rgba(217, 89, 38, 0.22)",
    QUIET_FILL: "rgba(57, 135, 229, 0.22)",
    **dict(zip(CATEGORICAL, CATEGORICAL_DARK, strict=True)),
    # sequential: dark near zero, bright at the top
    "#f3f8fe": "#1f2733",
    "#b7d3f6": "#1c4a85",
    "#6da7ec": "#2f78d0",
    "#256abf": "#6da7ec",
    "#0d366b": "#cde2fb",
    # diverging: dark neutral midpoint, poles brighten outward
    "#184f95": "#9ec5f4",
    "#ee8a86": "#b8403f",
    "#a8262a": "#f19b99",
}

TEMPLATE_NAME = "stockml"


def date_axes(fig: go.Figure) -> go.Figure:
    """Readable date hover and tick formats for every x-axis of a time-series figure."""
    return fig.update_xaxes(
        hoverformat="%a %d %b %Y",
        tickformatstops=[
            {"dtickrange": [None, 86_400_000 * 45], "value": "%d %b\n%Y"},
            {"dtickrange": [86_400_000 * 45, 86_400_000 * 365], "value": "%b\n%Y"},
            {"dtickrange": [86_400_000 * 365, None], "value": "%Y"},
        ],
    )


def model_color(name: str) -> str:
    """Stable colour for a model name (benchmark grey for unknown names)."""
    return MODEL_COLORS.get(name, BENCHMARK_COLOR)


def _build_template() -> go.layout.Template:
    axis = {
        "showgrid": True,
        "showspikes": True,
        "spikemode": "across",
        "spikesnap": "cursor",
        "spikethickness": 1,
        "spikedash": "dot",
        "spikecolor": TEXT_MUTED,
        "gridcolor": GRID,
        "gridwidth": 1,
        "zeroline": False,
        "showline": True,
        "linecolor": AXIS_LINE,
        "ticks": "outside",
        "tickcolor": AXIS_LINE,
        "ticklen": 4,
        "automargin": True,
        "title": {"font": {"size": 12, "color": TEXT_SECONDARY}, "standoff": 10},
        "tickfont": {"size": 11, "color": TEXT_MUTED},
    }
    return go.layout.Template(
        layout={
            "font": {"family": FONT_FAMILY, "size": 12, "color": TEXT_PRIMARY},
            "title": {
                "font": {"size": 15, "color": TEXT_PRIMARY, "weight": 600},
                "subtitle": {"font": {"size": 12, "color": TEXT_SECONDARY}},
                "x": 0.0,
                "xanchor": "left",
                "xref": "container",
                "y": 1.0,
                "yref": "container",
                "yanchor": "top",
                "pad": {"t": 14, "l": 8},
            },
            "paper_bgcolor": SURFACE,
            "plot_bgcolor": SURFACE,
            "colorway": list(CATEGORICAL),
            "margin": {"l": 64, "r": 24, "t": 92, "b": 48},
            "barcornerradius": 4,
            "bargap": 0.25,
            "hovermode": "x unified",
            "hoverlabel": {
                "bgcolor": SURFACE_RAISED,
                "bordercolor": AXIS_LINE,
                "align": "left",
                "font": {"family": FONT_FAMILY, "size": 12, "color": TEXT_PRIMARY},
            },
            "modebar": {
                "bgcolor": "rgba(0,0,0,0)",
                "color": AXIS_LINE,
                "activecolor": TEXT_PRIMARY,
            },
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.0,
                "xanchor": "left",
                "x": 0.0,
                "font": {"size": 11, "color": TEXT_SECONDARY},
                "bgcolor": "rgba(0,0,0,0)",
            },
            "xaxis": {**axis, "showgrid": False},
            # Value axes lean on the hairline grid alone: no spine, no tick marks.
            "yaxis": {**axis, "showline": False, "ticks": ""},
        }
    )


pio.templates[TEMPLATE_NAME] = _build_template()


def apply_theme(
    fig: go.Figure,
    title: str,
    x_title: str | None = None,
    y_title: str | None = None,
    height: int = 420,
    subtitle: str | None = None,
    legend_per_row: int = 3,
) -> go.Figure:
    """Apply the shared template, title and axis labels to ``fig`` and return it.

    Args:
        fig: Figure to style (modified in place and returned for chaining).
        title: Chart title.
        x_title: Label for the first x-axis, including units.
        y_title: Label for the first y-axis, including units.
        height: Figure height in pixels.
        subtitle: One-line "how to read this" hint shown under the title.
        legend_per_row: Legend entries expected to fit on one row (3 suits a half-width card);
            used to reserve enough top margin for a wrapping horizontal legend.
    """
    fig.update_layout(template=TEMPLATE_NAME, title_text=title, height=height)
    if subtitle:
        fig.update_layout(title_subtitle_text=subtitle)
    legend_items = sum(
        1 for trace in fig.data if trace.showlegend is not False and trace.visible is not False
    )
    has_legend = fig.layout.showlegend is not False and legend_items > 1
    legend_rows = math.ceil(legend_items / legend_per_row) if has_legend else 0
    fig.update_layout(
        margin_t=60 + (22 if subtitle else 0) + 22 * legend_rows + 8 * bool(legend_rows)
    )
    if x_title is not None:
        fig.update_layout(xaxis_title_text=x_title)
    if y_title is not None:
        fig.update_layout(yaxis_title_text=y_title)
    return fig
